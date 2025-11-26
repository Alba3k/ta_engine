# services/ta_engine/app.py
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import os

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import talib
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import OHLCV, Symbol, Timeframe
from .indicators.registry import INDICATORS

app = FastAPI(title="TA Engine Service")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Pydantic модели ---
class IndicatorRequest(BaseModel):
    indicator: str
    period: Optional[int] = 14
    backtrack: Optional[int] = 0
    results: Optional[int] = None
    addResultTimestamp: Optional[bool] = False


class Construct(BaseModel):
    symbol: str
    interval: str
    indicators: List[IndicatorRequest]


class BulkRequest(BaseModel):
    secret: str
    constructs: List[Construct]


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(BASE_DIR, "static", "favicon.ico"))


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/indicator/{name}")
def get_indicator(
    name: str,
    symbol: str = Query(..., description="Symbol, e.g. BTC/USDT"),
    interval: str = Query(..., description="Timeframe, e.g. 1h"),
    period: int = Query(14, description="Indicator length"),
    backtrack: int = Query(0, description="Candles back"),
    chart: str = Query("candles", description="candles or heikinashi"),
    addResultTimestamp: bool = Query(False),
    fromTimestamp: Optional[int] = None,
    toTimestamp: Optional[int] = None,
    results: Optional[str] = None,
):
    session: Session = SessionLocal()
    try:
        # --- получаем данные ---
        sym = session.query(Symbol).filter_by(name=symbol).first()
        tf = session.query(Timeframe).filter_by(code=interval).first()
        if not sym or not tf:
            return {"error": "Invalid symbol or timeframe"}

        q = session.query(OHLCV).filter_by(symbol_id=sym.id, timeframe_id=tf.id)
        if fromTimestamp:
            q = q.filter(OHLCV.start_time >= pd.to_datetime(fromTimestamp, unit="s"))
        if toTimestamp:
            q = q.filter(OHLCV.start_time < pd.to_datetime(toTimestamp, unit="s"))
        rows = q.order_by(OHLCV.start_time.asc()).all()
        if not rows:
            return {"error": "No data"}

        df = pd.DataFrame(
            [
                {
                    "open": float(r.open),
                    "high": float(r.high),
                    "low": float(r.low),
                    "close": float(r.close),
                    "volume": float(r.volume),
                    "start_time": r.start_time,
                }
                for r in rows
            ]
        )

        # --- Heikin Ashi ---
        if chart == "heikinashi":
            ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
            ha_open = (df["open"].shift(1) + df["close"].shift(1)) / 2
            ha_high = df[["high", "open", "close"]].max(axis=1)
            ha_low = df[["low", "open", "close"]].min(axis=1)
            df["open"], df["high"], df["low"], df["close"] = (
                ha_open.fillna(df["open"]),
                ha_high,
                ha_low,
                ha_close,
            )

        # --- выбираем индикатор ---
        fn = INDICATORS.get(name.lower())
        if not fn:
            return {"error": f"Indicator {name} not supported"}

        values = fn(df, period)

        # --- если индикатор вернул dict ---
        if isinstance(values, dict):
            idx = -1 - backtrack if backtrack > 0 else -1
            if results:
                if results == "max":
                    out = {
                        k: v.tolist() if isinstance(v, pd.Series) else [v]
                        for k, v in values.items()
                    }
                else:
                    n = int(results)
                    out = {
                        k: v.tail(n).tolist() if isinstance(v, pd.Series) else [v]
                        for k, v in values.items()
                    }
                return {"values": out}
            else:
                out = {}
                for k, v in values.items():
                    if isinstance(v, pd.Series):
                        out[k] = float(v.iloc[idx])
                    else:
                        out[k] = None if v is None else float(v)
                if addResultTimestamp:
                    ts = int(df.iloc[idx]["start_time"].timestamp())
                    return {"values": out, "timestamp": ts}
                return {"values": out}

        # --- если индикатор вернул Series ---
        if results:
            if results == "max":
                return {"values": values.tolist()}
            else:
                n = int(results)
                return {"values": values.tail(n).tolist()}

        idx = -1 - backtrack if backtrack > 0 else -1
        value = float(values.iloc[idx])
        if addResultTimestamp:
            ts = int(df.iloc[idx]["start_time"].timestamp())
            return {"value": value, "timestamp": ts}
        return {"value": value}

    finally:
        session.close()


# --- Bulk эндпойнт ---
@app.post("/bulk")
def bulk_indicators(req: BulkRequest):
    session: Session = SessionLocal()
    output = []

    try:
        for construct in req.constructs:
            sym = session.query(Symbol).filter_by(name=construct.symbol).first()
            tf = session.query(Timeframe).filter_by(code=construct.interval).first()
            if not sym or not tf:
                for ind in construct.indicators:
                    output.append(
                        {
                            "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                            "result": {},
                            "errors": ["Invalid symbol or timeframe"],
                        }
                    )
                continue

            rows = (
                session.query(OHLCV)
                .filter_by(symbol_id=sym.id, timeframe_id=tf.id)
                .order_by(OHLCV.start_time.asc())
                .all()
            )
            if not rows:
                for ind in construct.indicators:
                    output.append(
                        {
                            "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                            "result": {},
                            "errors": ["No data"],
                        }
                    )
                continue

            df = pd.DataFrame(
                [
                    {
                        "open": float(r.open),
                        "high": float(r.high),
                        "low": float(r.low),
                        "close": float(r.close),
                        "volume": float(r.volume),
                        "start_time": r.start_time,
                    }
                    for r in rows
                ]
            )

            for ind in construct.indicators:
                fn = INDICATORS.get(ind.indicator.lower())
                if not fn:
                    output.append(
                        {
                            "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                            "result": {},
                            "errors": [f"Indicator {ind.indicator} not supported"],
                        }
                    )
                    continue

                values = fn(df, ind.period)

                # dict индикаторы (каналы, облака)
                if isinstance(values, dict):
                    idx = -1 - ind.backtrack if ind.backtrack > 0 else -1
                    # result = {k: float(v.iloc[idx]) for k, v in values.items()}
                    result = {}
                    for k, v in values.items():
                        if isinstance(v, pd.Series):
                            result[k] = float(v.iloc[idx])
                        else:
                            # если это уже float или None
                            result[k] = None if v is None else float(v)

                    if ind.addResultTimestamp:
                        ts = int(df.iloc[idx]["start_time"].timestamp())
                        output.append(
                            {
                                "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                                "result": {"values": result, "timestamp": ts},
                                "errors": [],
                            }
                        )
                    else:
                        output.append(
                            {
                                "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                                "result": {"values": result},
                                "errors": [],
                            }
                        )
                else:
                    # Series индикаторы
                    idx = -1 - ind.backtrack if ind.backtrack > 0 else -1
                    value = float(values.iloc[idx])
                    if ind.addResultTimestamp:
                        ts = int(df.iloc[idx]["start_time"].timestamp())
                        output.append(
                            {
                                "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                                "result": {"value": value, "timestamp": ts},
                                "errors": [],
                            }
                        )
                    else:
                        output.append(
                            {
                                "id": f"{construct.symbol}_{construct.interval}_{ind.indicator}_{ind.period}_{ind.backtrack}",
                                "result": {"value": value},
                                "errors": [],
                            }
                        )

        return {"data": output}
    finally:
        session.close()

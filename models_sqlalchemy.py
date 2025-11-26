from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

class Symbol(Base):
    __tablename__ = "quotes_symbol"
    id = Column(Integer, primary_key=True)
    name = Column(String(20), unique=True)
    active = Column(Boolean, nullable=False)


class Timeframe(Base):
    __tablename__ = "quotes_timeframe"
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True)


class OHLCV(Base):
    __tablename__ = "quotes_ohlcv"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id", "timeframe_id", "start_time", name="unique_ohlcv_entry"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("quotes_symbol.id"), nullable=False)
    timeframe_id = Column(Integer, ForeignKey("quotes_timeframe.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class Signal(Base):
    __tablename__ = "signals_signal"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "timeframe_id",
            "strategy",
            "entry_time",
            name="unique_signal_entry",
        ),
    )

    id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey("quotes_symbol.id"), nullable=False)
    timeframe_id = Column(Integer, ForeignKey("quotes_timeframe.id"), nullable=False)
    strategy = Column(String(30), nullable=False)
    signal_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime)
    exit_price = Column(Float)
    pnl_pct = Column(Float)
    status = Column(String(10), default="active")
    created_at = Column(DateTime)
    closed_at = Column(DateTime)

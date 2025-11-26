# services/ta_engine/indicators/registry.py
import talib
import pandas as pd
import numpy as np


# --- Oscillators ---
def accosc(df, period=None):
    """Accelerator Oscillator (AO - simplified)"""
    return (df["high"] + df["low"]) / 2 - (
        (df["high"].shift(1) + df["low"].shift(1)) / 2
    )


def chop(df, period=14):
    """Choppiness Index"""
    tr = talib.TRANGE(df["high"], df["low"], df["close"])
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    return (
        100
        * np.log10(tr.rolling(period).sum() / atr.rolling(period).sum())
        / np.log10(period)
    )


def cmo(df, period=14):
    return talib.CMO(df["close"], timeperiod=period)


def coppockcurve(df, wma1=10, wma2=14, wma3=11):
    roc1 = talib.ROCR(df["close"], timeperiod=wma1)
    roc2 = talib.ROCR(df["close"], timeperiod=wma2)
    return talib.WMA(roc1 + roc2, timeperiod=wma3)


def dpo(df, period=20):
    """Detrended Price Oscillator"""
    sma = talib.SMA(df["close"], timeperiod=period)
    return df["close"] - sma.shift(int(period / 2) + 1)


def eom(df, period=14):
    """Ease of Movement"""
    distance = ((df["high"] + df["low"]) / 2).diff()
    box_ratio = df["volume"] / (df["high"] - df["low"])
    eom_raw = distance / box_ratio
    return eom_raw.rolling(period).mean()


def fosc(df, period=14):
    """Forecast Oscillator"""
    ma = talib.SMA(df["close"], timeperiod=period)
    return (df["close"] - ma) / ma * 100


def kvo(df, short=34, long=55, signal=13):
    """Klinger Volume Oscillator"""
    vf = (
        (df["high"] - df["low"]) - abs(df["high"].shift(1) - df["low"].shift(1))
    ) * df["volume"]
    short_ema = vf.ewm(span=short).mean()
    long_ema = vf.ewm(span=long).mean()
    return short_ema - long_ema


def vosc(df, short=14, long=28):
    """Volume Oscillator"""
    short_ma = df["volume"].rolling(short).mean()
    long_ma = df["volume"].rolling(long).mean()
    return (short_ma - long_ma) / long_ma * 100


# --- Volume Indicators ---
def ad(df, period=None):
    return talib.AD(df["high"], df["low"], df["close"], df["volume"])


def adosc(df, period=None):
    return talib.ADOSC(
        df["high"], df["low"], df["close"], df["volume"], fastperiod=3, slowperiod=10
    )


def obv(df, period=None):
    return talib.OBV(df["close"], df["volume"])


def cmf(df, period=20):
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    mf_vol = mf_mult * df["volume"]
    return mf_vol.rolling(period).sum() / df["volume"].rolling(period).sum()


def nvi(df, period=None):
    nvi = pd.Series(index=df.index, dtype="float64")
    nvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if df["volume"].iloc[i] < df["volume"].iloc[i - 1]:
            nvi.iloc[i] = (
                nvi.iloc[i - 1]
                + (df["close"].iloc[i] - df["close"].iloc[i - 1])
                / df["close"].iloc[i - 1]
                * nvi.iloc[i - 1]
            )
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi


def pvi(df, period=None):
    pvi = pd.Series(index=df.index, dtype="float64")
    pvi.iloc[0] = 1000
    for i in range(1, len(df)):
        if df["volume"].iloc[i] > df["volume"].iloc[i - 1]:
            pvi.iloc[i] = (
                pvi.iloc[i - 1]
                + (df["close"].iloc[i] - df["close"].iloc[i - 1])
                / df["close"].iloc[i - 1]
                * pvi.iloc[i - 1]
            )
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return pvi


def vwap(df, period=None):
    return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()


def vwma(df, period=20):
    return (df["close"] * df["volume"]).rolling(period).sum() / df["volume"].rolling(
        period
    ).sum()


# --- Overlap Studies ---
def accbands(df, period=20):
    """Acceleration Bands"""
    mid = talib.SMA(df["close"], timeperiod=period)
    upper = df["high"] * (1 + 4 * (df["high"] - df["low"]) / (df["high"] + df["low"]))
    lower = df["low"] * (1 - 4 * (df["high"] - df["low"]) / (df["high"] + df["low"]))
    return {"upper": upper, "middle": mid, "lower": lower}


def dema(df, period=20):
    return talib.DEMA(df["close"], timeperiod=period)


def hma(df, period=20):
    """Hull Moving Average"""
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = talib.WMA(df["close"], timeperiod=half_length)
    wma_full = talib.WMA(df["close"], timeperiod=period)
    diff = 2 * wma_half - wma_full
    return talib.WMA(diff, timeperiod=sqrt_length)


def kama(df, period=10):
    return talib.KAMA(df["close"], timeperiod=period)


def kdj(df, period=14):
    """KDJ indicator"""
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
    j = 3 * slowk - 2 * slowd
    return {"k": slowk, "d": slowd, "j": j}


def ma(df, period=20):
    return talib.MA(df["close"], timeperiod=period)


def mama(df, period=None):
    mama, fama = talib.MAMA(df["close"])
    return {"mama": mama, "fama": fama}


def tema(df, period=20):
    return talib.TEMA(df["close"], timeperiod=period)


def trima(df, period=20):
    return talib.TRIMA(df["close"], timeperiod=period)


def wma(df, period=20):
    return talib.WMA(df["close"], timeperiod=period)


def zlema(df, period=20):
    """Zero-Lag EMA"""
    lag = int((period - 1) / 2)
    return talib.EMA(
        df["close"] + (df["close"] - df["close"].shift(lag)), timeperiod=period
    )


def midpoint(df, period=14):
    return talib.MIDPOINT(df["close"], timeperiod=period)


def midprice(df, period=14):
    return talib.MIDPRICE(df["high"], df["low"], timeperiod=period)


def t3(df, period=5):
    return talib.T3(df["close"], timeperiod=period)


def vidya(df, period=14):
    """Variable Index Dynamic Average (simplified)"""
    return df["close"].ewm(span=period).mean()


def wilders(df, period=14):
    """Wilders Smoothing"""
    return df["close"].ewm(alpha=1 / period).mean()


def ht_trendline(df, period=None):
    return talib.HT_TRENDLINE(df["close"])


# --- Price Indicators ---
def avgprice(df, period=None):
    return talib.AVGPRICE(df["open"], df["high"], df["low"], df["close"])


def medprice(df, period=None):
    return talib.MEDPRICE(df["high"], df["low"])


def price(df, period=None):
    return df["close"]


def tr(df, period=None):
    return talib.TRANGE(df["high"], df["low"], df["close"])


def typprice(df, period=None):
    return talib.TYPPRICE(df["high"], df["low"], df["close"])


def wclprice(df, period=None):
    return talib.WCLPRICE(df["high"], df["low"], df["close"])


# --- Volatility Indicators ---
def atr(df, period=14):
    """Average True Range"""
    return talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)


def bbands(df, period=20):
    """Bollinger Bands"""
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=period)
    return {"upper": upper, "middle": middle, "lower": lower}


def bbw(df, period=20):
    """Bollinger Band Width"""
    upper, middle, lower = talib.BBANDS(df["close"], timeperiod=period)
    return (upper - lower) / middle


def donchianchannels(df, period=20):
    """Donchian Channels"""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return {"upper": upper, "lower": lower}


def keltnerchannels(df, period=20):
    """Keltner Channels"""
    ema = talib.EMA(df["close"], timeperiod=period)
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    upper = ema + 2 * atr
    lower = ema - 2 * atr
    return {"upper": upper, "middle": ema, "lower": lower}


def mass(df, period=25):
    """Mass Index (Donald Dorsey)"""
    hl_range = df["high"] - df["low"]
    ema1 = hl_range.ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    ratio = ema1 / ema2
    mass_index = ratio.rolling(window=period).sum()
    return mass_index


def natr(df, period=14):
    """Normalized ATR"""
    return talib.NATR(df["high"], df["low"], df["close"], timeperiod=period)


def squeeze(df, period=20):
    """Squeeze Momentum Indicator (SMI)"""
    # Simplified implementation: BB vs KC
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"], timeperiod=period)
    ema = talib.EMA(df["close"], timeperiod=period)
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    kc_upper = ema + 1.5 * atr
    kc_lower = ema - 1.5 * atr
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return squeeze_on.astype(int)


def stddev(df, period=20):
    """Standard Deviation"""
    return talib.STDDEV(df["close"], timeperiod=period)


def volatility(df, period=30):
    """Annualized Historical Volatility"""
    log_returns = np.log(df["close"] / df["close"].shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


# --- Trend / Momentum Indicators ---
def ema(df, period=14):
    return talib.EMA(df["close"], timeperiod=period)


def sma(df, period=14):
    return talib.SMA(df["close"], timeperiod=period)


def smma(df, period=14):
    """Smoothed Moving Average"""
    return df["close"].ewm(alpha=1 / period).mean()


def adx(df, period=14):
    return talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)


def adxr(df, period=14):
    return talib.ADXR(df["high"], df["low"], df["close"], timeperiod=period)


def ao(df, period=None):
    """Awesome Oscillator"""
    return (df["high"] + df["low"]) / 2 - (
        (df["high"].shift(1) + df["low"].shift(1)) / 2
    )


def dx(df, period=14):
    return talib.DX(df["high"], df["low"], df["close"], timeperiod=period)


def dm(df, period=14):
    """Directional Movement"""
    plus_dm = talib.PLUS_DM(df["high"], df["low"], timeperiod=period)
    minus_dm = talib.MINUS_DM(df["high"], df["low"], timeperiod=period)
    return {"plus_dm": plus_dm, "minus_dm": minus_dm}


def dmi(df, period=14):
    """Directional Movement Index"""
    plus_di = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    minus_di = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    adx_val = talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)
    return {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val}


def psar(df, period=None):
    """Parabolic SAR"""
    return talib.SAR(df["high"], df["low"])


def qstick(df, period=14):
    """Qstick"""
    return (df["close"] - df["open"]).rolling(period).mean()


def roc(df, period=14):
    """Rate of Change"""
    return talib.ROC(df["close"], timeperiod=period)


def rocp(df, period=10):
    return talib.ROCP(df["close"], timeperiod=period)


def rocr(df, period=10):
    return talib.ROCR(df["close"], timeperiod=period)


def rocr100(df, period=10):
    return talib.ROCR100(df["close"], timeperiod=period)


def supertrend(df, period=10, multiplier=3):
    """Supertrend"""
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    return {"upper": upperband, "lower": lowerband}


def vortex(df, period=14):
    """Vortex Indicator"""
    vi_plus = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    vi_minus = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
    return {"vi_plus": vi_plus, "vi_minus": vi_minus}


def williamsalligator(df, period=None):
    """Williams Alligator"""
    jaw = df["close"].shift(8).rolling(13).mean()
    teeth = df["close"].shift(5).rolling(8).mean()
    lips = df["close"].shift(3).rolling(5).mean()
    return {"jaw": jaw, "teeth": teeth, "lips": lips}


# --- Momentum Indicators ---
def apo(df, fast=12, slow=26):
    return talib.APO(df["close"], fastperiod=fast, slowperiod=slow)


def aroon(df, period=14):
    aroondown, aroonup = talib.AROON(df["high"], df["low"], timeperiod=period)
    return {"aroon_up": aroonup, "aroon_down": aroondown}


def aroonosc(df, period=14):
    return talib.AROONOSC(df["high"], df["low"], timeperiod=period)


def bop(df, period=None):
    return talib.BOP(df["open"], df["high"], df["low"], df["close"])


def cci(df, period=14):
    return talib.CCI(df["high"], df["low"], df["close"], timeperiod=period)


def macd(df, period=None):
    macd, signal, hist = talib.MACD(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    return {"macd": macd, "signal": signal, "hist": hist}


def macdext(df, period=None):
    macd, signal, hist = talib.MACDEXT(
        df["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    return {"macd": macd, "signal": signal, "hist": hist}


def marketfi(df, period=None):
    """Market Facilitation Index"""
    return (df["high"] - df["low"]) / df["volume"]


def mfi(df, period=14):
    return talib.MFI(
        df["high"], df["low"], df["close"], df["volume"], timeperiod=period
    )


def minus_di(df, period=14):
    return talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)


def minus_dm(df, period=14):
    return talib.MINUS_DM(df["high"], df["low"], timeperiod=period)


def mom(df, period=10):
    return talib.MOM(df["close"], timeperiod=period)


def plus_di(df, period=14):
    return talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)


def plus_dm(df, period=14):
    return talib.PLUS_DM(df["high"], df["low"], timeperiod=period)


def ppo(df, fast=12, slow=26, signal=9):
    return talib.PPO(df["close"], fastperiod=fast, slowperiod=slow, matype=0)


def price_di(df, period=None):
    """Price Direction"""
    return df["close"].diff()


def rsi(df, period=14):
    return talib.RSI(df["close"], timeperiod=period)


def rvgi(df, period=14):
    """Relative Vigor Index"""
    close_open = df["close"] - df["open"]
    high_low = df["high"] - df["low"]
    numerator = close_open.rolling(period).sum()
    denominator = high_low.rolling(period).sum()
    return numerator / denominator


def stc(df, period=10):
    """Schaff Trend Cycle (simplified)"""
    return talib.STOCHRSI(df["close"], timeperiod=period)[0]


def stoch(df, period=None):
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
    return {"slowk": slowk, "slowd": slowd}


def stochf(df, period=None):
    fastk, fastd = talib.STOCHF(df["high"], df["low"], df["close"])
    return {"fastk": fastk, "fastd": fastd}


def stochrsi(df, period=14):
    fastk, fastd = talib.STOCHRSI(df["close"], timeperiod=period)
    return {"fastk": fastk, "fastd": fastd}


def trix(df, period=30):
    return talib.TRIX(df["close"], timeperiod=period)


def ultosc(df, period=None):
    return talib.ULTOSC(df["high"], df["low"], df["close"])


def wad(df, period=None):
    """Williams Accumulation/Distribution"""
    return talib.AD(df["high"], df["low"], df["close"], df["volume"])


def willr(df, period=14):
    return talib.WILLR(df["high"], df["low"], df["close"], timeperiod=period)


def ichimoku(df, period=None):
    """
    Ichimoku Cloud (универсальный метод)
    Defaults: tenkan=9, kijun=26, senkou_b=52, displacement=26
    """

    # --- параметры ---
    tenkan = 9
    kijun = 26
    senkou_b = 52
    displacement = 26

    # если period передан как dict — переопределим
    if isinstance(period, dict):
        tenkan = int(period.get("tenkan", tenkan))
        kijun = int(period.get("kijun", kijun))
        senkou_b = int(period.get("senkou_b", senkou_b))
        displacement = int(period.get("displacement", displacement))

    # --- проверка достаточной длины данных ---
    min_len = max(tenkan, kijun, senkou_b) + displacement
    if len(df) < min_len:
        return {
            "tenkan_sen": None,
            "kijun_sen": None,
            "senkou_span_a": None,
            "senkou_span_b": None,
            "chikou_span": None,
            "error": f"Not enough data for Ichimoku (need >= {min_len} bars)"
        }

    # --- расчёт линий ---
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(displacement)
    chikou_span = close.shift(-displacement)

    # --- безопасная нормализация (NaN → None) ---
    def safe_float(x):
        if x is None:
            return None
        if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
            return None
        return float(x)

    # берём последнюю валидную точку для каждой линии
    def last_valid(series):
        last_idx = series.last_valid_index()
        if last_idx is None:
            return None
        return safe_float(series.loc[last_idx])

    return {
        "tenkan_sen": last_valid(tenkan_sen),
        "kijun_sen": last_valid(kijun_sen),
        "senkou_span_a": last_valid(senkou_span_a),
        "senkou_span_b": last_valid(senkou_span_b),
        "chikou_span": last_valid(chikou_span),
    }


# --- Pattern Recognition ---
def doji(df, period=None):
    return talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])


def hammer(df, period=None):
    return talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])


def engulfing(df, period=None):
    return talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"])


def tasukigap(df, period=None):
    return talib.CDLTASUKIGAP(df["open"], df["high"], df["low"], df["close"])


def thrusting(df, period=None):
    return talib.CDLTHRUSTING(df["open"], df["high"], df["low"], df["close"])


def tristar(df, period=None):
    return talib.CDLTRISTAR(df["open"], df["high"], df["low"], df["close"])


def upsidegap2crows(df, period=None):
    return talib.CDLUPSIDEGAP2CROWS(df["open"], df["high"], df["low"], df["close"])


def unique3river(df, period=None):
    return talib.CDLUNIQUE3RIVER(df["open"], df["high"], df["low"], df["close"])


def xsidegap3methods(df, period=None):
    return talib.CDLXSIDEGAP3METHODS(df["open"], df["high"], df["low"], df["close"])


# --- Math Operators ---
def math_sum(df, period=14):
    return df["close"].rolling(period).sum()


def math_max(df, period=14):
    return df["close"].rolling(period).max()


def math_maxindex(df, period=14):
    return df["close"].rolling(period).apply(lambda x: x.argmax())


def math_min(df, period=14):
    return df["close"].rolling(period).min()


def math_minindex(df, period=14):
    return df["close"].rolling(period).apply(lambda x: x.argmin())


def math_minmax(df, period=14):
    return {
        "min": df["close"].rolling(period).min(),
        "max": df["close"].rolling(period).max(),
    }


def math_minmaxindex(df, period=14):
    return {
        "min_index": df["close"].rolling(period).apply(lambda x: x.argmin()),
        "max_index": df["close"].rolling(period).apply(lambda x: x.argmax()),
    }


# --- Registry ---
INDICATORS = {
    # Oscillators (9)
    "accosc": accosc, "chop": chop, "cmo": cmo, "coppockcurve": coppockcurve,
    "dpo": dpo, "eom": eom, "fosc": fosc, "kvo": kvo, "vosc": vosc,

    # Volume (8)
    "ad": ad, "adosc": adosc, "obv": obv, "cmf": cmf, "nvi": nvi, "pvi": pvi,
    "vwap": vwap, "vwma": vwma,

    # Overlap Studies (17)
    "accbands": accbands, "dema": dema, "hma": hma, "ht_trendline": ht_trendline,
    "kama": kama, "kdj": kdj, "ma": ma, "mama": mama, "midpoint": midpoint,
    "midprice": midprice, "t3": t3, "tema": tema, "trima": trima,
    "vidya": vidya, "wilders": wilders, "wma": wma, "zlema": zlema,

    # Price (6)
    "avgprice": avgprice, "medprice": medprice, "price": price,
    "tr": tr, "typprice": typprice, "wclprice": wclprice,

    # Volatility (10)
    "atr": atr, "bbands": bbands, "bbw": bbw,
    "donchianchannels": donchianchannels, "keltnerchannels": keltnerchannels,
    "mass": mass, "natr": natr, "squeeze": squeeze, "stddev": stddev, 
    "volatility": volatility,

    # Trend / Momentum (18)
    "ema": ema, "sma": sma, "smma": smma, "adx": adx, "adxr": adxr, "ao": ao, 
    "dm": dm, "dmi": dmi, "psar": psar, "qstick": qstick, "supertrend": supertrend,  
    "dx": dx, "roc": roc, "vortex": vortex, "williamsalligator": williamsalligator,
    "rocp": rocp, "rocr": rocr, "rocr100": rocr100,

    # Momentum (27)
    "apo": apo, "aroon":aroon, "aroonosc":aroonosc, "bop":bop, "cci":cci,
    "ichimoku": ichimoku, "macd": macd, "macdext": macdext, "marketfi": marketfi, 
    "mfi": mfi, "minus_di": minus_di, "minus_dm": minus_dm, "mom": mom,
    "plus_di": plus_di, "plus_dm": plus_dm, "ppo": ppo, "price_di": price_di, 
    "rsi": rsi, "rvgi": rvgi, "stc": stc, "stoch": stoch, "stochf": stochf, 
    "stochrsi": stochrsi, "trix": trix, "ultosc": ultosc, "wad": wad, "willr": willr,

    # Pattern Recognition (9)
    "doji": doji, "hammer": hammer, "engulfing": engulfing, "tasukigap": tasukigap,
    "thrusting": thrusting, "tristar": tristar, "upsidegap2crows": upsidegap2crows, 
    "unique3river": unique3river, "xsidegap3methods": xsidegap3methods,

    # Math Operators (7)
    "sum": math_sum, "max": math_max ,
    "maxindex": math_maxindex, "min": math_min, "minindex": math_minindex, 
    "minmax": math_minmax, "minmaxindex": math_minmaxindex,
}
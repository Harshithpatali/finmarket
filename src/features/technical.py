import numpy as np
import pandas as pd

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]
    out["return_1d"] = c.pct_change(1)
    out["return_5d"] = c.pct_change(5)
    out["return_20d"] = c.pct_change(20)
    out["sma_20"] = c.rolling(20).mean()
    out["sma_50"] = c.rolling(50).mean()
    out["sma_200"] = c.rolling(200).mean()
    out["ema_20"] = c.ewm(span=20, adjust=False).mean()
    out["dist_sma20"] = (c - out["sma_20"]) / out["sma_20"]
    out["dist_sma50"] = (c - out["sma_50"]) / out["sma_50"]
    out["dist_sma200"] = (c - out["sma_200"]) / out["sma_200"]
    out["rsi_14"] = rsi(c)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    out["bb_position"] = (c - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / mid
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr_14"] / c
    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["volatility_20"] = out["return_1d"].rolling(20).std()
    out["volume_ratio"] = v / v.rolling(20).mean()
    out["volume_change"] = v.pct_change()
    out["return_z20"] = (
        (out["return_1d"] - out["return_1d"].rolling(20).mean())
        / out["return_1d"].rolling(20).std()
    )
    return out

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_20d",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width", "atr_pct",
    "volatility_10", "volatility_20",
    "volume_ratio", "volume_change", "return_z20",
    "market_return_1d", "market_return_5d",
    "relative_return_5d", "market_volatility_20",
]

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1"] = out["close"].pct_change()
    out["log_return_1"] = (out["close"].pct_change() + 1).apply(lambda x: pd.NA if x <= 0 else np.log(x))
    out["high_low_spread"] = (out["high"] - out["low"]) / out["close"]
    out["close_open_spread"] = (out["close"] - out["open"]) / out["open"]
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a default set of technical features.
    Keep this pure: the caller handles I/O and splitting.
    """
    feature_df = add_base_features(df)

    for window in (10, 20, 50):
        feature_df[f"sma_{window}"] = sma(feature_df["close"], window)
        feature_df[f"ema_{window}"] = ema(feature_df["close"], window)

    feature_df["rsi_14"] = rsi(feature_df["close"], window=14)

    bb_low, bb_mid, bb_high = bollinger_bands(feature_df["close"], window=20, num_std=2.0)
    feature_df["bb_low_20_2"] = bb_low
    feature_df["bb_mid_20_2"] = bb_mid
    feature_df["bb_high_20_2"] = bb_high

    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df

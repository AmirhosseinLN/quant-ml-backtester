from __future__ import annotations

import pandas as pd


def reshape_ohlcv_to_long(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yahoo-style wide OHLCV data with MultiIndex columns into long format.

    Output columns:
    date, ticker, open, high, low, close, volume
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns for Yahoo OHLCV data.")

    long_frames = []

    level_0 = prices.columns.get_level_values(0).unique()

    required_fields = {"Open", "High", "Low", "Close", "Volume"}
    available_fields = set(level_0)

    missing_fields = required_fields - available_fields
    if missing_fields:
        raise ValueError(f"Missing required OHLCV fields: {missing_fields}")

    tickers = prices["Close"].columns

    for ticker in tickers:
        df_ticker = pd.DataFrame({
            "date": prices.index,
            "ticker": ticker,
            "open": prices["Open"][ticker],
            "high": prices["High"][ticker],
            "low": prices["Low"][ticker],
            "close": prices["Close"][ticker],
            "volume": prices["Volume"][ticker],
        })
        long_frames.append(df_ticker)

    out = pd.concat(long_frames, axis=0, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out.groupby("ticker")["close"].pct_change(1)
    out["ret_5d"] = out.groupby("ticker")["close"].pct_change(5)
    out["ret_20d"] = out.groupby("ticker")["close"].pct_change(20)
    out["ret_60d"] = out.groupby("ticker")["close"].pct_change(60)
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    g = out.groupby("ticker")["close"]

    ma_20 = g.transform(lambda x: x.rolling(20).mean())
    ma_50 = g.transform(lambda x: x.rolling(50).mean())
    ma_100 = g.transform(lambda x: x.rolling(100).mean())

    out["ma_ratio_20"] = out["close"] / ma_20 - 1
    out["ma_ratio_50"] = out["close"] / ma_50 - 1
    out["ma_ratio_100"] = out["close"] / ma_100 - 1
    out["ma_cross_20_50"] = ma_20 / ma_50 - 1

    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    daily_ret = out.groupby("ticker")["close"].pct_change()

    out["vol_20d"] = (
        daily_ret.groupby(out["ticker"])
        .transform(lambda x: x.rolling(20).std())
    )

    out["vol_60d"] = (
        daily_ret.groupby(out["ticker"])
        .transform(lambda x: x.rolling(60).std())
    )

    return out


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    g = out.groupby("ticker")["volume"]
    vol_mean_20 = g.transform(lambda x: x.rolling(20).mean())
    vol_std_20 = g.transform(lambda x: x.rolling(20).std())

    out["volume_z_20"] = (out["volume"] - vol_mean_20) / vol_std_20
    out["volume_ratio_20"] = out["volume"] / vol_mean_20

    return out


def prepare_macro_features(macro: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    out = macro.copy()
    out = out.sort_index()
    out = out.loc[start_date:end_date].copy()

    out["term_spread"] = out["10y_treasury"] - out["2y_treasury"]

    return out


def merge_price_and_macro(price_long: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily asset data with macro data by date.
    Macro is forward-filled to daily timestamps after aligning to asset dates.
    """
    out = price_long.copy()

    macro_daily = macro.reindex(out["date"].sort_values().unique()).sort_index()
    macro_daily = macro_daily.ffill()

    out = out.merge(
        macro_daily,
        left_on="date",
        right_index=True,
        how="left",
    )

    return out.sort_values(["ticker", "date"]).reset_index(drop=True)

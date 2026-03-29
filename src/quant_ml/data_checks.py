from __future__ import annotations

import pandas as pd


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
    })
    summary["name"] = name
    return summary[["name", "dtype", "missing_count", "missing_pct"]]


def check_index_properties(df: pd.DataFrame) -> dict:
    return {
        "index_type": type(df.index).__name__,
        "is_monotonic_increasing": df.index.is_monotonic_increasing,
        "has_duplicates": df.index.has_duplicates,
        "n_rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
    }


def missing_by_column(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def missing_by_row(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum(axis=1).sort_values(ascending=False)


def flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns from yfinance into single strings like:
    Close_SPY, Volume_QQQ, etc.
    """
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            f"{col0}_{col1}" if col1 else str(col0)
            for col0, col1 in out.columns
        ]

    return out


def duplicate_dates(df: pd.DataFrame) -> pd.Index:
    return df.index[df.index.duplicated()]


def date_gaps(df: pd.DataFrame) -> pd.TimedeltaIndex:
    if len(df.index) < 2:
        return pd.to_timedelta([])

    diffs = df.index.to_series().diff().dropna()
    return diffs[diffs > pd.Timedelta(days=5)]

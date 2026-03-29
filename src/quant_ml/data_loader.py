from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .config import RAW_DATA_DIR


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_yahoo_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance using yfinance.
    Returns a pandas DataFrame with multi-index columns.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
    )
    return data


def save_yahoo_data(
    data: pd.DataFrame,
    filename: str = "yahoo_prices.parquet",
) -> Path:
    """
    Save Yahoo Finance data to data/raw/ as parquet.
    """
    ensure_directory(RAW_DATA_DIR)
    output_path = RAW_DATA_DIR / filename
    data.to_parquet(output_path)
    return output_path


def download_fred_series(
    series_map: Dict[str, str],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download multiple macro series from FRED.
    series_map example: {'fed_funds': 'FEDFUNDS'}
    """
    fred = Fred(api_key=api_key)
    frames = []

    for feature_name, series_id in series_map.items():
        series = fred.get_series(series_id)
        df = series.to_frame(name=feature_name)
        df.index.name = "date"
        frames.append(df)

    macro = pd.concat(frames, axis=1).sort_index()
    return macro


def save_fred_data(
    data: pd.DataFrame,
    filename: str = "fred_macro.parquet",
) -> Path:
    """
    Save FRED macro data to data/raw/ as parquet.
    """
    ensure_directory(RAW_DATA_DIR)
    output_path = RAW_DATA_DIR / filename
    data.to_parquet(output_path)
    return output_path

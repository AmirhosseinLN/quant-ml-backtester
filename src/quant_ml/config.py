from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "EEM"]
BENCHMARK_TICKER = "SPY"

START_DATE = "2012-01-01"
END_DATE = "2025-12-31"

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "10y_treasury": "GS10",
    "2y_treasury": "GS2",
}

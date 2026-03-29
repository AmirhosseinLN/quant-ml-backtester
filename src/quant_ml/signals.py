from __future__ import annotations

import pandas as pd


def make_weekly_top_n_signals(
    df: pd.DataFrame,
    score_col: str = "proba_up",
    top_n: int = 2,
) -> pd.DataFrame:
    """
    Build weekly top-N signals from model scores.
    Ranking happens once per week on the last available trading day.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["week"] = out["date"].dt.to_period("W")

    weekly = (
        out.sort_values("date")
        .groupby(["ticker", "week"])
        .tail(1)
        .copy()
    )

    weekly["rank"] = (
        weekly.groupby("week")[score_col]
        .rank(ascending=False, method="first")
    )

    weekly["signal"] = (weekly["rank"] <= top_n).astype(int)

    signals = weekly[["date", "ticker", "signal"]].copy()

    out = out.merge(signals, on=["date", "ticker"], how="left")
    out["signal"] = out.groupby("ticker")["signal"].ffill().fillna(0)

    return out


def equal_weight_from_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["weight"] = (
        out.groupby("date")["signal"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )

    return out

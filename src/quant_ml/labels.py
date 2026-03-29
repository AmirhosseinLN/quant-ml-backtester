from __future__ import annotations

import pandas as pd


def add_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = df.copy()
    out[f"fwd_ret_{horizon}d"] = (
        out.groupby("ticker")["close"].shift(-horizon) / out["close"] - 1
    )
    return out


def add_direction_label(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    out = add_forward_returns(df, horizon=horizon)
    out[f"target_up_{horizon}d"] = (out[f"fwd_ret_{horizon}d"] > 0).astype("float")
    return out

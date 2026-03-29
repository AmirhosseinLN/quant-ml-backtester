from __future__ import annotations

import pandas as pd


def compute_momentum_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly momentum strategy:
    - compute ranking once per week (last available day)
    - hold positions for the entire next week
    """
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"])
    out["week"] = out["date"].dt.to_period("W")

    # pick last trading day of each week per ticker
    weekly = (
        out.sort_values("date")
        .groupby(["ticker", "week"])
        .tail(1)
    )

    # rank cross-sectionally per week
    weekly["rank"] = (
        weekly.groupby("week")["ret_60d"]
        .rank(ascending=False, method="first")
    )

    weekly["signal"] = (weekly["rank"] <= 2).astype(int)

    # now expand signals back to daily
    signals = weekly[["date", "ticker", "signal"]].copy()

    # merge back to full dataset
    out = out.merge(signals, on=["date", "ticker"], how="left")

    # forward-fill signals within each ticker
    out["signal"] = out.groupby("ticker")["signal"].ffill()

    # fill initial NaNs with 0
    out["signal"] = out["signal"].fillna(0)

    return out

def equal_weight_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert signals into equal-weight portfolio.
    """
    out = df.copy()

    weights = (
        out.groupby(["date"])["signal"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )

    out["weight"] = weights

    return out


def compute_strategy_returns(df: pd.DataFrame, cost_per_trade: float = 0.001) -> pd.DataFrame:
    """
    Compute daily portfolio returns with correct cross-sectional alignment.
    """
    out = df.copy()

    # daily returns
    out["ret_1d"] = out.groupby("ticker")["close"].pct_change()

    # pivot weights into matrix (date x ticker)
    weight_matrix = out.pivot(index="date", columns="ticker", values="weight")

    # shift entire portfolio (this is the KEY fix)
    weight_shifted = weight_matrix.shift(1)

    # pivot returns
    ret_matrix = out.pivot(index="date", columns="ticker", values="ret_1d")

    # portfolio return = sum(weight * return)
    portfolio_ret = (weight_shifted * ret_matrix).sum(axis=1)

    portfolio = portfolio_ret.to_frame(name="gross_return")

    # turnover = sum absolute change in weights
    turnover = weight_matrix.diff().abs().sum(axis=1)

    portfolio["cost"] = turnover * cost_per_trade
    portfolio["net_return"] = portfolio["gross_return"] - portfolio["cost"]

    portfolio["cum_return"] = (1 + portfolio["net_return"]).cumprod()

    return portfolio
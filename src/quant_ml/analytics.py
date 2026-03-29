from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(portfolio: pd.DataFrame) -> Dict[str, float]:
    r = portfolio["net_return"].dropna()

    if r.empty or r.std() == 0:
        sharpe = float("nan")
        annual_return = float("nan")
        annual_vol = float("nan")
    else:
        sharpe = float((r.mean() / r.std()) * np.sqrt(252))
        annual_return = float((1 + r.mean()) ** 252 - 1)
        annual_vol = float(r.std() * np.sqrt(252))

    cum = portfolio["cum_return"].dropna()
    if cum.empty:
        max_drawdown = float("nan")
    else:
        max_drawdown = float((cum / cum.cummax() - 1).min())

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def build_drawdown_series(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.copy()
    out = out.sort_index()
    out["rolling_peak"] = out["cum_return"].cummax()
    out["drawdown"] = out["cum_return"] / out["rolling_peak"] - 1
    return out


def portfolio_to_dashboard_frame(
    portfolio: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    out = portfolio.copy().reset_index().rename(columns={"index": "date"})
    out["strategy_name"] = strategy_name
    return out


def metrics_dict_to_frame(metrics: Dict[str, float], strategy_name: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy_name": strategy_name,
        **metrics,
    }])


def extract_logistic_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Expects a sklearn Pipeline with final step named 'clf'.
    """
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    out = pd.DataFrame({
        "feature": feature_cols,
        "importance": coefs,
        "abs_importance": np.abs(coefs),
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)

    return out


def extract_rf_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    importances = model.feature_importances_

    out = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
        "abs_importance": np.abs(importances),
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)

    return out
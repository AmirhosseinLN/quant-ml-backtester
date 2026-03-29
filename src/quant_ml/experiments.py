from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .baseline import (
    compute_momentum_signal,
    equal_weight_portfolio,
    compute_strategy_returns,
)
from .models import fit_predict_logistic, fit_predict_rf, time_split
from .signals import make_weekly_top_n_signals, equal_weight_from_signal


def compute_metrics(portfolio: pd.DataFrame) -> Dict[str, float]:
    r = portfolio["net_return"].dropna()

    if r.empty or r.std() == 0:
        sharpe = float("nan")
    else:
        sharpe = (r.mean() / r.std()) * (252 ** 0.5)

    max_dd = (portfolio["cum_return"] / portfolio["cum_return"].cummax() - 1).min()

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def run_baseline_on_test(
    test_df: pd.DataFrame,
    cost_per_trade: float = 0.001,
) -> pd.DataFrame:
    baseline_df = test_df.copy()
    baseline_df = compute_momentum_signal(baseline_df)
    baseline_df = equal_weight_portfolio(baseline_df)
    baseline_portfolio = compute_strategy_returns(
        baseline_df,
        cost_per_trade=cost_per_trade,
    )
    return baseline_portfolio


def run_ml_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
    top_n: int,
    cost_per_trade: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if model_name == "logistic":
        model, preds = fit_predict_logistic(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )
    elif model_name == "rf":
        model, preds = fit_predict_rf(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    ml_df = make_weekly_top_n_signals(
        preds,
        score_col="proba_up",
        top_n=top_n,
    )
    ml_df = equal_weight_from_signal(ml_df)

    ml_portfolio = compute_strategy_returns(
        ml_df,
        cost_per_trade=cost_per_trade,
    )

    return preds, ml_portfolio


def run_experiment_grid(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_end_date: str,
    experiment_grid: List[Dict],
    cost_per_trade: float = 0.001,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    split = time_split(df, train_end_date=train_end_date)
    train_df = split.train
    test_df = split.test

    baseline_portfolio = run_baseline_on_test(
        test_df=test_df,
        cost_per_trade=cost_per_trade,
    )
    baseline_metrics = compute_metrics(baseline_portfolio)

    results = []
    portfolios = {"baseline": baseline_portfolio}

    for exp in experiment_grid:
        model_name = exp["model_name"]
        top_n = exp["top_n"]

        preds, ml_portfolio = run_ml_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            model_name=model_name,
            top_n=top_n,
            cost_per_trade=cost_per_trade,
        )

        common_index = baseline_portfolio.index.intersection(ml_portfolio.index)
        baseline_aligned = baseline_portfolio.loc[common_index].copy()
        ml_aligned = ml_portfolio.loc[common_index].copy()

        baseline_aligned_metrics = compute_metrics(baseline_aligned)
        ml_aligned_metrics = compute_metrics(ml_aligned)

        exp_name = f"{model_name}_top{top_n}"

        results.append({
            "experiment_name": exp_name,
            "model_name": model_name,
            "top_n": top_n,
            "train_end_date": train_end_date,
            "baseline_sharpe": baseline_aligned_metrics["sharpe"],
            "ml_sharpe": ml_aligned_metrics["sharpe"],
            "baseline_max_drawdown": baseline_aligned_metrics["max_drawdown"],
            "ml_max_drawdown": ml_aligned_metrics["max_drawdown"],
            "sharpe_improvement": ml_aligned_metrics["sharpe"] - baseline_aligned_metrics["sharpe"],
            "drawdown_improvement": ml_aligned_metrics["max_drawdown"] - baseline_aligned_metrics["max_drawdown"],
            "n_test_rows": len(test_df),
            "n_pred_rows": len(preds),
            "avg_proba_up": float(preds["proba_up"].mean()),
            "std_proba_up": float(preds["proba_up"].std()),
        })

        portfolios[exp_name] = ml_aligned

    results_df = pd.DataFrame(results).sort_values(
        by=["ml_sharpe", "ml_max_drawdown"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return results_df, portfolios
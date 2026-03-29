from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainTestSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def time_split(
    df: pd.DataFrame,
    train_end_date: str,
) -> TrainTestSplit:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    train = out[out["date"] <= train_end_date].copy()
    test = out[out["date"] > train_end_date].copy()

    return TrainTestSplit(train=train, test=test)


def build_logistic_model() -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    return model


def fit_predict_logistic(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[Pipeline, pd.DataFrame]:
    model = build_logistic_model()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]

    model.fit(X_train, y_train)

    preds = test_df[["date", "ticker", "close", "fwd_ret_5d"]].copy()
    preds["proba_up"] = model.predict_proba(X_test)[:, 1]
    preds["pred_label"] = (preds["proba_up"] >= 0.5).astype(int)

    return model, preds


def build_rf_model() -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    return model


def fit_predict_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    model = build_rf_model()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]

    model.fit(X_train, y_train)

    preds = test_df[["date", "ticker", "close", "fwd_ret_5d"]].copy()
    preds["proba_up"] = model.predict_proba(X_test)[:, 1]
    preds["pred_label"] = (preds["proba_up"] >= 0.5).astype(int)

    return model, preds
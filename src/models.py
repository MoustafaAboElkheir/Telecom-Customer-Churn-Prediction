"""
Customer Churn Prediction — Model Definitions and Training Pipeline
====================================================================
Defines all classification models, training pipeline, and evaluation
utilities for the telecom customer churn prediction task.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              classification_report, confusion_matrix)
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)

# Model registry with tuned hyperparameters
MODEL_REGISTRY: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        C=0.5, max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3,   # accounts for ~25% churn rate
        random_state=42, eval_metric="logloss", verbosity=0,
    ),
}


def encode_features(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Label-encode all categorical features and return X, y."""
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col]
    return X, y


def load_and_split(data_path: str, test_size: float = 0.30):
    """
    Load churn data, encode features, apply 70/30 split,
    and balance the training set via undersampling.
    """
    df = pd.read_csv(data_path)
    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Balance training set (50-50 undersampling)
    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
    logger.info(f"Balanced train class dist: {np.bincount(y_train_bal)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_bal)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train_bal, y_test, scaler


def train_and_evaluate(model_name: str, X_train, y_train,
                       X_test, y_test) -> Dict[str, Any]:
    """Train a single model and return metrics and the fitted model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODEL_REGISTRY[model_name]
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, proba), 4),
    }
    logger.info(f"  AUC-ROC={metrics['auc_roc']} | F1={metrics['f1']} | Recall={metrics['recall']}")
    return {"model": model, "metrics": metrics}


def run_all_models(data_path: str) -> Dict[str, Dict]:
    """Train and evaluate all models; return a comparison dictionary."""
    X_train, X_test, y_train, y_test, _ = load_and_split(data_path)
    results = {}
    for name in MODEL_REGISTRY:
        result = train_and_evaluate(name, X_train, y_train, X_test, y_test)
        results[name] = result["metrics"]
    return results

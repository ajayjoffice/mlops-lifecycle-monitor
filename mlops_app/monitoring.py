"""Dependency-light drift and performance monitoring."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_app.config import DRIFT_THRESHOLD, MIN_ACCURACY, PRODUCTION_PATH, REFERENCE_PATH
from mlops_app.data import TARGET


def validate_frame(frame: pd.DataFrame, feature_names: list[str]) -> None:
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Features must be numeric and cannot contain missing values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features must contain only finite values")


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected.astype(float), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.astype(float), bins=edges)[0] / len(expected)
    a = np.histogram(actual.astype(float), bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, production: pd.DataFrame, feature_names: list[str],
                 threshold: float = DRIFT_THRESHOLD) -> dict:
    validate_frame(reference, feature_names)
    validate_frame(production, feature_names)
    scores = {name: _psi(reference[name], production[name]) for name in feature_names}
    drifted = [name for name, value in scores.items() if value >= threshold]
    return {"drift_detected": bool(drifted), "threshold": threshold,
            "drifted_features": drifted, "feature_psi": scores,
            "mean_psi": float(np.mean(list(scores.values()))),
            "reference_rows": len(reference), "production_rows": len(production)}


def performance_report(y_true, predictions, min_accuracy: float = MIN_ACCURACY) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = float(accuracy_score(y_true, predictions))
    return {"accuracy": accuracy, "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "min_accuracy": min_accuracy, "performance_degraded": accuracy < min_accuracy}


def get_monitoring_snapshot(model, feature_names: list[str], reference_path: Path = REFERENCE_PATH,
                            production_path: Path = PRODUCTION_PATH) -> dict:
    if not reference_path.exists() or not production_path.exists():
        return {"ready": False, "message": "Run training and generate demo datasets first."}
    reference, production = pd.read_csv(reference_path), pd.read_csv(production_path)
    drift = drift_report(reference, production, feature_names)
    x_prod = production[feature_names]
    predictions = model.predict(x_prod)
    snapshot = {"ready": True, "drift": drift,
                "prediction_counts": pd.Series(predictions).value_counts().sort_index().to_dict(),
                "prediction_rate": float(np.mean(predictions)), "rows_scored": len(predictions)}
    if TARGET in production:
        snapshot["performance"] = performance_report(production[TARGET], predictions)
    snapshot["retraining_recommended"] = bool(
        drift["drift_detected"] or snapshot.get("performance", {}).get("performance_degraded", False)
    )
    return snapshot


def load_training_metadata(model_path: Path) -> dict:
    path = model_path.with_suffix(".metrics.json")
    return json.loads(path.read_text()) if path.exists() else {}

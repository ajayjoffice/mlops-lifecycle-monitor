"""Reproducible training and MLflow experiment logging."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlops_app.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_PATH
from mlops_app.data import split_dataset


def train_model(model_path: Path = MODEL_PATH, random_state: int = 42, n_estimators: int = 250) -> dict:
    x_train, x_test, y_train, y_test = split_dataset(random_state=random_state)
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=n_estimators, max_depth=6, random_state=random_state),
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        mlflow.log_params({"random_state": random_state, "n_estimators": n_estimators,
                           "max_depth": 6, "dataset": "sklearn breast_cancer"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=x_train.head(2),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="export")
        result = {"run_id": run.info.run_id, "model_path": str(model_path), "metrics": metrics,
                  "feature_names": list(x_train.columns), "rows_train": len(x_train),
                  "rows_test": len(x_test)}
    model_path.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate_predictions(y_true: pd.Series, predictions) -> dict:
    """Small helper used by tests and external evaluation scripts."""
    return {"accuracy": float(accuracy_score(y_true, predictions)),
            "f1": float(f1_score(y_true, predictions, zero_division=0))}

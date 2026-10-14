from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACT_DIR / "model.joblib"))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", DATA_DIR / "reference.csv"))
PRODUCTION_PATH = Path(os.getenv("PRODUCTION_PATH", DATA_DIR / "production.csv"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "breast-cancer-classification")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.20"))
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.90"))

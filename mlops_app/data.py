"""Dataset loading, deterministic splitting, and demo data generation."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "target"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def split_dataset(test_size: float = 0.2, random_state: int = 42):
    x, y = load_dataset()
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)


def write_demo_datasets(reference_path, production_path, random_state: int = 42) -> None:
    x_train, x_test, y_train, y_test = split_dataset(random_state=random_state)
    reference = x_train.copy()
    reference[TARGET] = y_train
    # Deliberately perturb one feature to make the sample drift report demonstrable.
    production = x_test.copy()
    production["mean radius"] = production["mean radius"] * 1.22
    production[TARGET] = y_test.to_numpy()
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(reference_path, index=False)
    production.to_csv(production_path, index=False)

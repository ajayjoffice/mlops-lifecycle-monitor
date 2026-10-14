import numpy as np
import pandas as pd
import pytest

from mlops_app.monitoring import drift_report, performance_report, validate_frame


def test_detects_shifted_feature():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    production = pd.DataFrame({"x": rng.normal(4, 1, 1000)})
    report = drift_report(reference, production, ["x"])
    assert report["drift_detected"]
    assert "x" in report["drifted_features"]


def test_validation_rejects_missing_and_null_features():
    with pytest.raises(ValueError, match="Missing"):
        validate_frame(pd.DataFrame({"y": [1]}), ["x"])
    with pytest.raises(ValueError, match="missing"):
        validate_frame(pd.DataFrame({"x": [None]}), ["x"])


def test_performance_threshold():
    report = performance_report([0, 1, 1, 0], [0, 0, 1, 0], min_accuracy=0.9)
    assert report["performance_degraded"]
    assert report["accuracy"] == 0.75

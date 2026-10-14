from fastapi.testclient import TestClient

from mlops_app import api


def test_health_and_invalid_payload():
    api.MODEL = None
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["status"] == "ok"
        bad_payload = {"features": {"unexpected": 1}}
        assert client.post("/predict", json=bad_payload).status_code == 422
        assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_prediction_returns_service_unavailable_without_model(monkeypatch):
    monkeypatch.setattr(api, "MODEL", None)
    monkeypatch.setattr(api, "FEATURE_NAMES", ["x"])
    from fastapi import HTTPException
    import pytest
    with pytest.raises(HTTPException) as error:
        api._predict([{"x": 1.0}])
    assert error.value.status_code == 503

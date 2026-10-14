# Production ML Lifecycle & Model Monitoring

**Version 1 is complete.** This beginner-friendly project demonstrates the basic lifecycle of a machine learning model: train it, serve predictions, and monitor sample production data. Future updates are planned, including logging real API predictions and connecting other AI projects to the dashboard.

> This is an educational project using a public dataset. It is not a medical tool and must not be used to make health decisions.

## What the project does

1. **Learns from examples:** Trains a classifier on the public Wisconsin Diagnostic Breast Cancer dataset included with scikit-learn. It classifies measurements as benign or malignant and saves the trained model.
2. **Records the training run:** Uses MLflow to track the training settings and evaluation scores so runs can be inspected and compared.
3. **Answers requests:** A FastAPI service accepts the 30 numeric measurements and returns a class prediction and estimated probability for class 1.
4. **Checks for changes:** Compares sample production measurements with reference training data using Population Stability Index (PSI), and flags features that have shifted past a configurable threshold.
5. **Shows the results:** A Streamlit dashboard displays API health, prediction counts, feature drift, and evaluation metrics for the labeled sample production data.
6. **Makes the project repeatable:** Docker Compose starts the services together. Pytest covers core behavior, and GitHub Actions runs tests and a compile check on pushes and pull requests.

## Project flow

```text
Public dataset → train.py → saved model + MLflow run
                         ↓
                  FastAPI predictions
                         ↓
Reference/sample production data → drift + evaluation → Streamlit dashboard
```

Monitoring currently compares the bundled sample CSV files. The production sample has a deliberate shift in `mean radius` so the drift check is easy to see. Its labels come from the held-out dataset split and enable a sample evaluation; this is not live production accuracy.

## Run with Docker Compose

Requires Docker Engine with the Compose plugin.

```bash
docker compose up --build
```

Open these local pages while the Compose command is running:

- Dashboard: [http://localhost:8501](http://localhost:8501)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow experiment UI: [http://localhost:5001](http://localhost:5001)

Compose trains the model in its setup service before starting the API. The host port for MLflow is 5001 because 5000 may already be in use.

## Run without Docker

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python train.py
```

Then start the API and dashboard in separate terminals (activate the same virtual environment in each):

```bash
uvicorn mlops_app.api:app --reload
streamlit run dashboard.py
```

For the MLflow UI outside Docker, run `mlflow ui --backend-store-uri sqlite:///mlflow.db`; it uses port 5000 by default.

Training writes the model to `artifacts/model.joblib`, metrics metadata alongside it, and MLflow runs to the local SQLite database `mlflow.db` with artifacts under `mlruns/`. The reference and sample production CSVs are in `data/`.

## Try a prediction

Use `GET /model` to see the required feature names. In the API docs, expand `POST /predict`, choose **Try it out**, and provide a JSON body in this shape:

```json
{
  "features": {
    "mean radius": 10.32,
    "mean texture": 16.35
  }
}
```

The example is shortened for readability. A real request must include all 30 feature names returned by `GET /model`. A complete example is in [`data/sample_prediction.json`](data/sample_prediction.json). Invalid, missing, non-numeric, or non-finite measurements return HTTP 422; if no trained model is loaded, prediction returns HTTP 503.

Other useful endpoints:

- `GET /health` — API health, model-loaded status, and in-memory prediction count.
- `GET /model` — model type and required feature names.
- `POST /predict/batch` — predict for multiple rows.
- `GET /monitoring` — drift, prediction distribution, sample performance, and retraining recommendation.

## Training and evaluation

The reproducible baseline uses a stratified 80/20 split with random state 42 and a `StandardScaler` plus `RandomForestClassifier` (`n_estimators=250`, `max_depth=6`). MLflow records accuracy, precision, recall, F1, ROC AUC, parameters, and the model artifact. Metrics are produced by the training run rather than hard-coded in this README.

## Monitoring settings

The dashboard recommends retraining when any feature's PSI meets `DRIFT_THRESHOLD` (default `0.20`) or sample accuracy falls below `MIN_ACCURACY` (default `0.90`). Set these environment variables to change the thresholds. The project only recommends retraining; it does not automatically train or deploy another model.

PSI is used here as a straightforward learning example, not as a universal drift test. Prediction counts are held in API memory and reset when the service restarts. The sample CSVs are static until regenerated by running `python train.py`.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

The tests cover deterministic data splitting, metric calculations, invalid feature data, drift detection, threshold behavior, and API error handling.

## V1 status and planned updates

**V1 is complete** with training, MLflow tracking, prediction serving, sample drift monitoring, dashboard, tests, and containerized startup. Future versions may add persistent logging for real API predictions, stronger monitoring, and integrations for other AI projects such as RAG and realtime systems. Those updates are planned, not included in V1.

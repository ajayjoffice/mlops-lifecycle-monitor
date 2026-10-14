"""Train and create the sample reference/production datasets."""
from mlops_app.config import MODEL_PATH, PRODUCTION_PATH, REFERENCE_PATH
from mlops_app.data import write_demo_datasets
from mlops_app.training import train_model

if __name__ == "__main__":
    result = train_model(MODEL_PATH)
    write_demo_datasets(REFERENCE_PATH, PRODUCTION_PATH)
    print(result)

"""Module that detects data drift using Evidently.

It can be scheduled to run periodically with prefect.
Drift detection is logged to MLFlow.
"""

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset

# import requests
from evidently.report import Report
from prefect import flow, task

from preprocess import Preprocesser
from settings import DATA_PATH, MLFLOW_TRACKING_URI

# TODO: Retrieve latest data from predict-API
# @task
# def collect_latest_predictions():
#     response = requests.get("http://localhost:8000/predict_logs")
#     return pd.DataFrame(response.json())


def simulate_drift(data: pd.DataFrame) -> pd.DataFrame:
    """Simulate data drift by multiplying the ShelfCapacity by a random number between 0.4 and 0.6.

    Also shuffles the data.
    Args:
        data (pd.DataFrame): The data to simulate drift on.

    Returns:
        pd.DataFrame: The data with simulated drift.
    """
    data["ShelfCapacity"] = data["ShelfCapacity"].apply(lambda x: x * np.random.uniform(0.4, 0.6))

    data = data.sample(frac=1).reset_index(drop=True)
    return data


@task
def load_ref_data() -> pd.DataFrame:
    """Prefect task that loads the reference data from the data source.

    Returns:
        pd.DataFrame: The reference data.
    """
    preprocesser = Preprocesser(file_path=DATA_PATH)
    ref_data, _, _, _ = preprocesser()
    return ref_data


@task
def detect_drift(new_data: pd.DataFrame, ref_data: pd.DataFrame) -> dict:
    """Prefect task that detects data drift using Evidently, and logs the drift metrics to MLFlow.

    Args:
        new_data (pd.DataFrame): The new data.
        ref_data (pd.DataFrame): The reference data.

    Returns:
        dict: The drift metrics.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data, current_data=new_data)

    drift_metrics = report.as_dict()["metrics"][0]["result"]

    with mlflow.start_run():
        mlflow.log_metrics(drift_metrics)

    return drift_metrics


@flow
def detect_drift_flow():
    """Prefect flow that detects data drift using Evidently."""
    ref_data = load_ref_data()
    new_data = simulate_drift(ref_data)
    detect_drift(new_data, ref_data)


if __name__ == "__main__":
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Drift detection")
    # Run every 5 minutes
    detect_drift_flow.serve(name="drift-detection", cron="*/10 * * * *")

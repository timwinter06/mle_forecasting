"""Prefect flow for training a random forest regression model."""

import logging
from typing import Tuple

import mlflow
import pandas as pd
from prefect import flow, task

from preprocess import Preprocesser
from settings import DATA_PATH, EXPERIMENT_NAME, MLFLOW_TRACKING_URI, N_ESTIMATORS
from train import RFModelTrainer, track_with_mlflow

logging.basicConfig(level=logging.INFO)
logging.info("Initializing train flow")


@task
def load_data() -> Tuple[pd.DataFrame]:
    """Prefect task for loading the data returning split train and test data.

    Returns:
        Tuple[pd.DataFrame]: The train features, train labels, test features, and test labels
    """
    logging.info("Loading data...")
    preprocesser = Preprocesser(file_path=DATA_PATH)
    train_x, train_y, test_x, test_y = preprocesser()
    return train_x, train_y, test_x, test_y


@task
def train_model(train_x: pd.DataFrame, train_y: pd.DataFrame, params: dict) -> RFModelTrainer:
    """Prefect task for training the model.

    Args:
        train_x (pd.DataFrame): The train features.
        train_y (pd.DataFrame): The train labels.
        params (dict): The model hyperparameters.

    Returns:
        RFModelTrainer: The trained model
    """
    logging.info("Training model...")
    model_trainer = RFModelTrainer(**params)
    model_trainer.train(train_x, train_y)
    return model_trainer


@task
def evaluate_model(model_trainer: RFModelTrainer, test_x: pd.DataFrame, test_y: pd.DataFrame) -> dict:
    """Prefect task for evaluating the model.

    Args:
        model_trainer (RFModelTrainer): The trained model.
        test_x (pd.DataFrame): The test features.
        test_y (pd.DataFrame): The test labels.

    Returns:
        dict: The metrics.
    """
    logging.info("Evaluating model...")
    metrics = model_trainer.evaluate(test_x, test_y)
    logging.info(f"Metrics: {metrics}")
    return metrics


@task
def track_model(model_trainer: RFModelTrainer, params: dict, metrics: dict, train_x: pd.DataFrame) -> None:
    """Prefect task for tracking the model with MLflow.

    Args:
        model_trainer (RFModelTrainer): The trained model.
        params (dict): The model hyperparameters.
        metrics (dict): The metrics.
    """
    logging.info("Tracking model with MLflow...")
    track_with_mlflow(model_trainer.model, params, metrics, train_x)


@flow
def train_flow():
    """Prefect flow for training a random forest regression model."""
    logging.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_x, train_y, test_x, test_y = load_data()
    params = {
        "n_estimators": N_ESTIMATORS,
        "max_features": round(len(train_x.columns) / 3),
        "max_depth": len(train_x.columns),
    }
    model_trainer = train_model(train_x, train_y, params)
    metrics = evaluate_model(model_trainer, test_x, test_y)
    track_model(model_trainer, params, metrics, train_x)


if __name__ == "__main__":
    # To run locally you need to run this cmd: prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api

    # Run once the firt time
    train_flow()

    # Schedule to run every Sunday at 00:00
    train_flow.serve(name="train-flow", cron="0 0 * * 0")

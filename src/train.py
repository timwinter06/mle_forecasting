"""Module for training, evaluating and saving a random forest model."""

import logging
import math
import os
import pickle
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import Preprocesser
from settings import DATA_PATH, EXPERIMENT_NAME

logging.basicConfig(level=logging.INFO)


class RFModelTrainer:
    """Class for training a random forest model."""

    def __init__(self, max_features: int, max_depth: int, n_estimators: int = 100) -> None:
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth
        )

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train the model.

        Args:
            x_train (pd.DataFrame): The training data.
            y_train (pd.DataFrame): The training labels.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predict the labels for the test data.

        Args:
            x_test (pd.DataFrame): The test data.

        Returns:
            np.ndarray: The predicted labels.
        """
        return self.model.predict(x_test)

    def predict_units(self, x_test: pd.DataFrame) -> int:
        """Predict the units by coverting the log to units.

        Args:
            x_test (pd.DataFrame): The test data.

        Returns:
            int: The predicted units.
        """
        return int(math.exp(self.predict(x_test)[0]))

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            x_test (pd.DataFrame): The test data.
            y_test (pd.DataFrame): The test labels.

        Returns:
            Dict[str, float]: The metrics.
        """
        predictions = self.predict(x_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
        }
        return metrics

    def save(self, save_path: str) -> None:
        """Save the model to disk.

        Args:
            save_path (str): The path to save the model to.
        """
        with open(save_path, "wb") as file:
            pickle.dump(self.model, file)


def track_with_mlflow(
    model: RFModelTrainer, params: Dict[str, int], metrics: Dict[str, float], x_train: pd.DataFrame
) -> None:
    """Track the model with MLflow.

    Args:
        model (RFModelTrainer): The model to track.
        params (Dict[str, int]): The parameters to track.
        metrics (Dict[str, float]): The metrics to track.
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("Forecast model", "Random Forest")

        signature = infer_signature(x_train, model.predict(x_train))
        # Log model
        _ = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="forecast_model",
            signature=signature,
            registered_model_name="random_forest_regressor",
        )


if __name__ == "__main__":
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    logging.info(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logging.info("Loading data...")
    preprocesser = Preprocesser(file_path=DATA_PATH)
    train_x, train_y, test_x, test_y = preprocesser()

    logging.info("Training model...")
    params = {"n_estimators": 100, "max_features": round(len(train_x.columns) / 3), "max_depth": len(train_x.columns)}
    model_trainer = RFModelTrainer(
        n_estimators=params["n_estimators"], max_features=params["max_features"], max_depth=params["max_depth"]
    )
    model_trainer.train(train_x, train_y)

    logging.info("Evaluating model...")
    metrics = model_trainer.evaluate(test_x, test_y)
    print(metrics)

    logging.info("Tracking model with MLflow...")
    track_with_mlflow(model_trainer.model, params, metrics, train_x)

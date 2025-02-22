"""Module for training, evaluating and saving a random forest model."""

import math
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import Preprocesser
from settings import DATA_PATH, MODEL_PATH


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


if __name__ == "__main__":
    # Load train/test data
    preprocesser = Preprocesser(file_path=DATA_PATH)
    train_x, train_y, test_x, test_y = preprocesser()

    # Train model
    model_trainer = RFModelTrainer(
        n_estimators=100, max_features=round(len(train_x.columns) / 3), max_depth=len(train_x.columns)
    )
    model_trainer.train(train_x, train_y)

    # Evaluate model
    metrics = model_trainer.evaluate(test_x, test_y)
    print(metrics)

    # Save model
    model_trainer.save(MODEL_PATH)

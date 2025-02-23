"""Module for making predictions."""

import math
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from settings import MODEL_PATH


class ModelPredictor:
    """Class for making predictions"""

    def __init__(self, model_path: str) -> None:
        """Initialize the predictor.

        Args:
            model_path (str): The path to the model.
        """
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predict the labels for the test data.

        Args:
            x_test (pd.DataFrame): The test data.

        Returns:
            np.ndarray: The predicted labels.
        """
        predictions = self.model.predict(x_test)
        return predictions

    def predict_units(self, x_test: pd.DataFrame) -> int:
        """Predict the units by converting the log to units.

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


if __name__ == "__main__":
    columns = [
        "StoreCount",
        "ShelfCapacity",
        "PromoShelfCapacity",
        "IsPromo",
        "ItemNumber",
        "CategoryCode",
        "GroupCode",
        "month",
        "weekday",
        "UnitSales_-7",
        "UnitSales_-14",
        "UnitSales_-21",
    ]

    custom_example = pd.DataFrame(
        data=[
            (781, 12602.000, 4922, True, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075),
        ],
        columns=columns,
    )
    model = ModelPredictor(MODEL_PATH)
    units = model.predict_units(custom_example)

    print(f"Model predicts {units} units.")

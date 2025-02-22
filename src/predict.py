"""Module for making predictions."""

import math
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import add_lagged_feature_to_df, create_train_test_split, preprocess, split_x_y


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
        return int(math.exp(self.predict(x_test)))

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
    FILE_PATH = "./data/raw/dataset.csv"
    df = pd.read_csv(FILE_PATH, sep=";", header=0)
    df_cleaned = preprocess(df)
    train_df, test_df = create_train_test_split(df_cleaned)
    NUMBER_OF_LAGS = [7, 14, 21]  # 1 week ago, 2 weeks ago, 3 weeks ago
    FEATURE_TO_LAG = "UnitSales"
    train_df_lag = add_lagged_feature_to_df(train_df, NUMBER_OF_LAGS, FEATURE_TO_LAG)
    test_df_lag = add_lagged_feature_to_df(test_df, NUMBER_OF_LAGS, FEATURE_TO_LAG)
    train_x, train_y = split_x_y(train_df_lag)
    test_x, test_y = split_x_y(test_df_lag)

    model = ModelPredictor("./models/forecasting_model.pkl")

    metrics = model.evaluate(test_x, test_y)
    print(metrics)

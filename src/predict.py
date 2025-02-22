"""
This script should contain some function to run your model / make predictions.
By default: the api.py calls the make_predictions function here
"""

from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


class ModelPredictor:
    MODEL_FILE = "models/mymodel.pkl"

    def __init__(self) -> None:
        self.model = joblib.load(self.MODEL_FILE)

    def __call__(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        predictions = self.model.predict(data)
        return predictions

    def evaluate(self, features: Any, labels: Any) -> Dict[str, float]:
        predicted = self(features)

        # N.B. this is not generic, but model specific
        metrics = {
            "AUC": roc_auc_score(y_true=labels, y_score=predicted),
            "accuracy": accuracy_score(y_true=labels, y_pred=predicted),
        }
        return metrics

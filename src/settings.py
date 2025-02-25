"""Module for commonly used variables."""

import os

# Paths
DATA_PATH = "./data/raw/dataset.csv"
MODEL_PATH = "./models/forecasting_model.pkl"

# Hyperparameters
NUMBER_OF_LAGS = [7, 14, 21]
FEATURE_TO_LAG = "UnitSales"
N_ESTIMATORS = 100

# MLFlow settings
EXPERIMENT_NAME = "unit_sales_forecasting"
MODEL_NAME = "random_forest_regressor"
MODEL_VERSION = "latest"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")

# Data columns
COLUMNS = [
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

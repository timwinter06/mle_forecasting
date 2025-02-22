"""Module for commonly used variables."""

# Paths
DATA_PATH = "./data/raw/dataset.csv"
MODEL_PATH = "./models/forecasting_model.pkl"

# Hyperparameters
NUMBER_OF_LAGS = [7, 14, 21]
FEATURE_TO_LAG = "UnitSales"

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

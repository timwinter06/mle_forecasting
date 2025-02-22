"""Module for preprocessing the data."""

import datetime
from typing import Tuple

import numpy as np
import pandas as pd


def preprocess(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data before training.

    Args:
        df_prep (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """

    # Log transform the target column!
    df_prep["UnitSales"] = np.log(df_prep["UnitSales"])

    # convert the column DateKey (string) to a date column
    df_prep["DateKey"] = pd.to_datetime(df_prep["DateKey"], format="%Y%m%d")
    df_prep["month"] = df_prep["DateKey"].dt.month
    df_prep["weekday"] = df_prep["DateKey"].dt.weekday

    # Drop null values
    df_prep_clean_0 = df_prep[df_prep["UnitSales"].notnull()].copy()
    df_prep_clean = df_prep_clean_0[df_prep_clean_0["ShelfCapacity"].notnull()].copy()

    # Convert columns to correct format
    df_prep_clean["month"] = df_prep_clean["month"].astype("category")
    df_prep_clean["weekday"] = df_prep_clean["weekday"].astype("category")

    return df_prep_clean


def add_lagged_feature_to_df(input_df: pd.DataFrame, lag_iterator: object, feature: str) -> pd.DataFrame:
    """A function that will expand an input dataframe with lagged variables of a specified feature
    Note that the lag is calculated over time (datekey) but also kept appropriate over itemnumber (article)

    Args:
        input_df (pd.DataFrame): input dataframe that should contain the feature and itemnr.
        lag_iterator (object): an object that can be iterator over, that includes info about the requested nr of lags.
        feature (str): feature that we want to include the lag of in the dataset.

    Returns:
        pd.DataFrame: the expanded dataframe.
    """
    output_df = input_df.copy()
    for lag in lag_iterator:
        df_to_lag = input_df[["DateKey", "ItemNumber", feature]].copy()
        # we add the nr of days equal to the lag we want
        df_to_lag["DateKey"] = df_to_lag["DateKey"] + datetime.timedelta(days=lag)

        # the resulting dataframe contains sales data that is lag days old for the date that is in that row
        df_to_lag = df_to_lag.rename(columns={feature: feature + "_-" + str(lag)})

        # we join this dataframe on the original dataframe to add the lagged variable as feature
        output_df = output_df.merge(df_to_lag, how="left", on=["DateKey", "ItemNumber"])

    # Drop datekey column that is not used anymore
    output_df = output_df.drop(columns=["DateKey"])

    # drop na rows that have been caused by these lags
    return output_df.dropna()


def train_test_split(total_df: pd.DataFrame, tr_split_date: datetime.datetime) -> pd.DataFrame:
    """Split a dataframe into train and test sets based on a split date.

    Args:
        total_df (pd.DataFrame): The dataframe to split.
        tr_split_date (datetime.datetime): The date to split the dataframe on.

    Returns:
        pd.DataFrame: The train and test sets.
    """
    tr_df = total_df[total_df["DateKey"].dt.date <= tr_split_date].copy()
    tst_df = total_df[total_df["DateKey"].dt.date > tr_split_date].copy()
    return tr_df, tst_df


def create_train_test_split(
    df_to_split: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a train and test split of the data.

    Args:
        df_to_split (pd.DataFrame): The dataframe to split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test sets.
    """
    # We split the data in a train set and a test set, we do this, 80, 20 percent respectively.
    nr_of_unique_dates = len(df_to_split.DateKey.unique())
    train_split_delta = round(nr_of_unique_dates * 0.8)
    train_split_date = df_to_split.DateKey.dt.date.min() + datetime.timedelta(days=train_split_delta)
    train_df, test_df = train_test_split(df_to_split, train_split_date)
    train_df["GroupCode"] = train_df["GroupCode"].astype("category")
    train_df["ItemNumber"] = train_df["ItemNumber"].astype("category")
    train_df["CategoryCode"] = train_df["CategoryCode"].astype("category")

    # determine unique item numbers, and filter the validation and test on these
    items_we_train_on = train_df["ItemNumber"].unique()
    test_df_filtered = test_df[test_df["ItemNumber"].isin(items_we_train_on)].copy()
    test_df_filtered["GroupCode"] = test_df_filtered["GroupCode"].astype("category")
    test_df_filtered["ItemNumber"] = test_df_filtered["ItemNumber"].astype("category")
    test_df_filtered["CategoryCode"] = test_df_filtered["CategoryCode"].astype("category")

    return train_df, test_df_filtered


def split_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into X and y for training.

    Args:
        df (pd.DataFrame): The dataframe to split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The X and y.
    """
    df_x = df.drop(columns=["UnitSales"])
    df_y = df["UnitSales"]
    return df_x, df_y


if __name__ == "__main__":
    # Read in data
    FILE_PATH = "./data/raw/dataset.csv"
    df = pd.read_csv(FILE_PATH, sep=";", header=0)

    # Clean data
    df_cleaned = preprocess(df)

    # Split data
    train_df, test_df = create_train_test_split(df_cleaned)

    # Add lagged variables
    NUMBER_OF_LAGS = [7, 14, 21]  # 1 week ago, 2 weeks ago, 3 weeks ago
    FEATURE_TO_LAG = "UnitSales"

    # make the lags per dataset (no data leakage) and also do the NaN filtering per set
    train_df_lag = add_lagged_feature_to_df(train_df, NUMBER_OF_LAGS, FEATURE_TO_LAG)
    test_df_lag = add_lagged_feature_to_df(test_df, NUMBER_OF_LAGS, FEATURE_TO_LAG)

    # Split into x and y
    train_x, train_y = split_x_y(train_df_lag)
    test_x, test_y = split_x_y(test_df_lag)

    # Print shapes to check
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")

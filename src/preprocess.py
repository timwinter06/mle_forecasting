"""Module for preprocessing the data."""

import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from settings import DATA_PATH, FEATURE_TO_LAG, NUMBER_OF_LAGS


class Preprocesser:
    """Class for preprocessing the data."""

    def __init__(self, file_path: str) -> None:
        """Initialize the preprocessor.

        Args:
            file_path (str): The path to the data.
        """
        self.file_path = file_path
        self.df_data = pd.read_csv(file_path, sep=";", header=0)

    def __call__(
        self, number_of_lags: List[int] = NUMBER_OF_LAGS, feature_to_lag: str = FEATURE_TO_LAG
    ) -> Tuple[pd.DataFrame]:
        """Call the preprocessor to clean data add lagged features
        and split into train and test.

        Args:
            number_of_lags (List[int], optional): The number of lags to add. Defaults to [7, 14, 21].
            feature_to_lag (str, optional): The feature to lag. Defaults to "UnitSales".

        Returns:
            Tuple[pd.DataFrame]: The train features, train labels, test features, and test labels.
        """
        # Clean data
        df_cleaned = self.preprocess(self.df_data)

        # Split data
        train_df, test_df = self.create_train_test_split(df_cleaned)

        # Add lagged variables
        # make the lags per dataset (no data leakage) and also do the NaN filtering per set
        train_df_lag = self.add_lagged_feature_to_df(train_df, number_of_lags, feature_to_lag)
        test_df_lag = self.add_lagged_feature_to_df(test_df, number_of_lags, feature_to_lag)

        # Split into x and y
        train_x, train_y = self.split_x_y(train_df_lag)
        test_x, test_y = self.split_x_y(test_df_lag)

        return train_x, train_y, test_x, test_y

    def preprocess(self, df_prep: pd.DataFrame) -> pd.DataFrame:
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

    def add_lagged_feature_to_df(self, input_df: pd.DataFrame, lag_iterator: object, feature: str) -> pd.DataFrame:
        """A function that will expand an input dataframe with lagged variables of a specified feature
        Note that the lag is calculated over time (datekey) but also kept appropriate over itemnumber (article)

        Args:
            input_df (pd.DataFrame): input dataframe that should contain the feature and itemnr.
            lag_iterator (object): an object that can be iterator over, that includes info about the requested nr
            of lags.
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

    def train_test_split(self, total_df: pd.DataFrame, tr_split_date: datetime.datetime) -> pd.DataFrame:
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
        self,
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
        train_df, test_df = self.train_test_split(df_to_split, train_split_date)
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

    def split_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    # Read in the data
    preprocesser = Preprocesser(file_path=DATA_PATH)

    # Call the preprocessor
    train_x, train_y, test_x, test_y = preprocesser()

    # Print shapes to check
    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")

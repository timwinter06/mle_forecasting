from io import StringIO

import pandas as pd
import pytest

from preprocess import Preprocesser  # Assuming the module is named preprocesser.py

# Sample CSV data
CSV_DATA = """DateKey;StoreCount;ShelfCapacity;PromoShelfCapacity;IsPromo;ItemNumber;CategoryCode;GroupCode;UnitSales
20240101;5;50;10;1;1001;C1;G1;10
20240102;5;50;10;0;1001;C1;G1;15
20240103;5;50;10;1;1001;C1;G1;20
20240104;5;50;10;0;1001;C1;G1;25
20240105;5;50;10;1;1001;C1;G1;30
"""


@pytest.fixture(scope="module")
def sample_data():
    """Fixture to provide sample data for testing."""
    return StringIO(CSV_DATA)


@pytest.fixture(scope="module")
def preprocesser(sample_data: str) -> Preprocesser:
    """Fixture to provide a Preprocesser instance for testing.

    Args:
        sample_data (str): Sample data for testing.

    Returns:
        Preprocesser: A Preprocesser instance.
    """
    return Preprocesser(file_path=sample_data)


def test_preprocess(preprocesser: Preprocesser):
    """Test the preprocess method of the Preprocesser class.

    Args:
        preprocesser (Preprocesser): An instance of the Preprocesser class.
    """
    ""
    df = preprocesser.preprocess(pd.read_csv(StringIO(CSV_DATA), sep=";"))
    assert "DateKey" in df.columns
    assert "month" in df.columns
    assert "weekday" in df.columns
    assert not df.isnull().values.any()


def test_add_lagged_feature(preprocesser: Preprocesser):
    """Test the add_lagged_feature_to_df method of the Preprocesser class.

    Args:
        preprocesser (Preprocesser): An instance of the Preprocesser class.
    """
    df = preprocesser.preprocess(pd.read_csv(StringIO(CSV_DATA), sep=";"))
    lagged_df = preprocesser.add_lagged_feature_to_df(df, [1, 2], "UnitSales")
    assert "UnitSales_-1" in lagged_df.columns
    assert "UnitSales_-2" in lagged_df.columns


def test_preprocesser(preprocesser: Preprocesser):
    """Test the Preprocesser class main method.

    Args:
        preprocesser (Preprocesser): An instance of the Preprocesser class.
    """
    train_x, train_y, test_x, test_y = preprocesser()
    assert isinstance(train_x, pd.DataFrame)
    assert isinstance(train_y, pd.Series)
    assert isinstance(test_x, pd.DataFrame)
    assert isinstance(test_y, pd.Series)

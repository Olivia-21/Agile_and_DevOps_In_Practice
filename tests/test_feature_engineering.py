"""
test_feature_engineering.py - Unit Tests for Feature Engineering Module

Tests for:
- Creating date-derived features (Month, Day, Weekday, Season)
- Calculating Total Fare where missing
- Encoding categorical variables (one-hot and label)
- Scaling numerical features
- Splitting data into train/test sets
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    create_date_features,
    calculate_total_fare,
    encode_categorical_features,
    scale_numerical_features,
    split_data,
)


# -- Fixtures -----------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a sample cleaned DataFrame for testing."""
    return pd.DataFrame({
        "Airline": ["Biman", "US-Bangla", "Novoair", "Biman", "Regent",
                     "US-Bangla", "Novoair", "Biman", "Regent", "Biman"],
        "Source": ["Dhaka", "Dhaka", "Chattogram", "Dhaka", "Sylhet",
                   "Dhaka", "Chattogram", "Dhaka", "Sylhet", "Dhaka"],
        "Destination": ["Chattogram", "Cox's Bazar", "Dhaka", "Sylhet", "Dhaka",
                        "Cox's Bazar", "Dhaka", "Sylhet", "Dhaka", "Chattogram"],
        "Date": pd.to_datetime([
            "2024-01-15", "2024-06-20", "2024-03-10", "2024-10-05",
            "2024-12-25", "2024-07-15", "2024-04-01", "2024-11-10",
            "2024-02-14", "2024-08-20",
        ]),
        "Base Fare": [3500.0, 4200.0, 3100.0, 3800.0, 5000.0,
                      4100.0, 3200.0, 3900.0, 4800.0, 3600.0],
        "Tax & Surcharge": [500.0, 600.0, 450.0, 550.0, 700.0,
                            580.0, 460.0, 560.0, 680.0, 510.0],
        "Total Fare": [4000.0, 4800.0, 3550.0, 4350.0, 5700.0,
                       4680.0, 3660.0, 4460.0, 5480.0, 4110.0],
    })


@pytest.fixture
def df_missing_total():
    """Create a DataFrame with missing Total Fare values."""
    return pd.DataFrame({
        "Base Fare": [3500.0, 4200.0, 3100.0],
        "Tax & Surcharge": [500.0, 600.0, 450.0],
        "Total Fare": [4000.0, np.nan, np.nan],
    })


# -- create_date_features Tests ----------------------------------------

class TestCreateDateFeatures:
    """Tests for the create_date_features function."""

    def test_creates_month_feature(self, sample_df):
        """Test that Month feature is created correctly."""
        result = create_date_features(sample_df)
        assert "Month" in result.columns
        assert result["Month"].iloc[0] == 1  # January

    def test_creates_day_feature(self, sample_df):
        """Test that Day feature is created correctly."""
        result = create_date_features(sample_df)
        assert "Day" in result.columns
        assert result["Day"].iloc[0] == 15

    def test_creates_weekday_feature(self, sample_df):
        """Test that Weekday feature is created correctly."""
        result = create_date_features(sample_df)
        assert "Weekday" in result.columns
        # 2024-01-15 is a Monday (0)
        assert result["Weekday"].iloc[0] == 0

    def test_creates_season_feature(self, sample_df):
        """Test that Season feature is created correctly."""
        result = create_date_features(sample_df)
        assert "Season" in result.columns
        # January -> Winter
        assert result["Season"].iloc[0] == "Winter"

    def test_season_mapping(self, sample_df):
        """Test that all seasons are mapped correctly."""
        result = create_date_features(sample_df)
        seasons = result["Season"].unique()
        valid_seasons = {"Winter", "Summer", "Monsoon", "Autumn"}
        assert set(seasons).issubset(valid_seasons)

    def test_no_date_column_skips(self):
        """Test that function skips gracefully if no date column exists."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = create_date_features(df)
        assert "Month" not in result.columns


# -- calculate_total_fare Tests ----------------------------------------

class TestCalculateTotalFare:
    """Tests for the calculate_total_fare function."""

    def test_fills_missing_total_fare(self, df_missing_total):
        """Test that missing Total Fare values are calculated."""
        result = calculate_total_fare(df_missing_total)
        assert not result["Total Fare"].isna().any()

    def test_calculation_correct(self, df_missing_total):
        """Test that calculated Total Fare = Base Fare + Tax & Surcharge."""
        result = calculate_total_fare(df_missing_total)
        # Row 1: 4200 + 600 = 4800
        assert result["Total Fare"].iloc[1] == 4800.0
        # Row 2: 3100 + 450 = 3550
        assert result["Total Fare"].iloc[2] == 3550.0

    def test_existing_values_unchanged(self, df_missing_total):
        """Test that existing Total Fare values are not modified."""
        result = calculate_total_fare(df_missing_total)
        assert result["Total Fare"].iloc[0] == 4000.0


# -- encode_categorical_features Tests --------------------------------

class TestEncodeCategoricalFeatures:
    """Tests for the encode_categorical_features function."""

    def test_onehot_encoding(self):
        """Test one-hot encoding creates dummy columns."""
        df = pd.DataFrame({"Airline": ["Biman", "US-Bangla", "Novoair"], "Fare": [100, 200, 300]})
        result = encode_categorical_features(df, method="onehot")
        # One-hot with drop_first=True should create N-1 columns
        assert "Airline" not in result.columns
        assert result.shape[1] > 1

    def test_label_encoding(self):
        """Test label encoding converts strings to integers."""
        df = pd.DataFrame({"Airline": ["Biman", "US-Bangla", "Novoair"], "Fare": [100, 200, 300]})
        result = encode_categorical_features(df, method="label")
        assert result["Airline"].dtype in [np.int32, np.int64]

    def test_invalid_method_raises_error(self):
        """Test that an invalid encoding method raises ValueError."""
        df = pd.DataFrame({"Airline": ["Biman", "US-Bangla"]})
        with pytest.raises(ValueError):
            encode_categorical_features(df, method="invalid")

    def test_no_categorical_columns(self):
        """Test with no categorical columns returns unchanged."""
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        result = encode_categorical_features(df)
        assert list(result.columns) == ["A", "B"]


# -- scale_numerical_features Tests -----------------------------------

class TestScaleNumericalFeatures:
    """Tests for the scale_numerical_features function."""

    def test_returns_scaled_df_and_scaler(self):
        """Test that function returns a tuple of (DataFrame, scaler)."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
        result_df, scaler = scale_numerical_features(df)
        assert isinstance(result_df, pd.DataFrame)
        assert scaler is not None

    def test_scaled_values_have_zero_mean(self):
        """Test that scaled features have approximately zero mean."""
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result_df, _ = scale_numerical_features(df)
        assert abs(result_df["A"].mean()) < 1e-10

    def test_exclude_cols_not_scaled(self):
        """Test that excluded columns are not scaled."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "Total Fare": [4000.0, 4800.0, 3550.0],
        })
        result_df, _ = scale_numerical_features(df, exclude_cols=["Total Fare"])
        # Total Fare should remain unchanged
        assert result_df["Total Fare"].tolist() == [4000.0, 4800.0, 3550.0]


# -- split_data Tests -------------------------------------------------

class TestSplitData:
    """Tests for the split_data function."""

    def test_returns_four_arrays(self, sample_df):
        """Test that split_data returns X_train, X_test, y_train, y_test."""
        # Prepare numeric-only df
        df = sample_df[["Base Fare", "Tax & Surcharge", "Total Fare"]].copy()
        X_train, X_test, y_train, y_test = split_data(df, target_col="Total Fare")
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

    def test_split_ratio(self, sample_df):
        """Test that the 80/20 split ratio is approximately correct."""
        df = sample_df[["Base Fare", "Tax & Surcharge", "Total Fare"]].copy()
        X_train, X_test, y_train, y_test = split_data(df, target_col="Total Fare")
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.15  # Allow some tolerance

    def test_target_not_in_features(self, sample_df):
        """Test that the target column is not in X."""
        df = sample_df[["Base Fare", "Tax & Surcharge", "Total Fare"]].copy()
        X_train, X_test, _, _ = split_data(df, target_col="Total Fare")
        assert "Total Fare" not in X_train.columns

    def test_missing_target_raises_error(self):
        """Test that missing target column raises ValueError."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError):
            split_data(df, target_col="NonExistent")

    def test_reproducibility(self, sample_df):
        """Test that the same random_state produces the same split."""
        df = sample_df[["Base Fare", "Tax & Surcharge", "Total Fare"]].copy()
        X1, _, y1, _ = split_data(df, target_col="Total Fare", random_state=42)
        X2, _, y2, _ = split_data(df, target_col="Total Fare", random_state=42)
        pd.testing.assert_frame_equal(X1, X2)

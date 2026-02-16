"""
test_data_cleaner.py - Unit Tests for Data Cleaning Module

Tests for:
- Dropping irrelevant/unnamed columns
- Removing duplicate rows
- Handling missing values (numerical & categorical)
- Fixing invalid fare values
- Normalizing city names
- Converting data types
- Full cleaning pipeline
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaner import (
    drop_irrelevant_columns,
    remove_duplicates,
    handle_missing_values,
    fix_invalid_fares,
    normalize_city_names,
    convert_data_types,
    clean_dataset,
    get_cleaning_summary,
)


# -- Fixtures -----------------------------------------------------------

@pytest.fixture
def raw_df():
    """Create a sample raw DataFrame with known issues."""
    return pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3, 4],
        "Airline": ["Biman", "US-Bangla", "Novoair", "Biman", "Biman"],
        "Source": ["Dacca", "Dhaka", "chittagong", "Dhaka", "Dacca"],
        "Destination": ["Chittagong", "Cox's Bazar", "Dhaka", "Sylhet", "Chittagong"],
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-01-15"],
        "Base Fare": [3500.0, 4200.0, -100.0, 3800.0, 3500.0],
        "Tax & Surcharge": [500.0, 600.0, 450.0, 550.0, 500.0],
        "Total Fare": [4000.0, 4800.0, 3550.0, 4350.0, 4000.0],
    })


@pytest.fixture
def df_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        "Airline": ["Biman", None, "Novoair", "Biman"],
        "Source": ["Dhaka", "Dhaka", None, "Sylhet"],
        "Base Fare": [3500.0, np.nan, 3100.0, 3800.0],
        "Tax & Surcharge": [500.0, 600.0, np.nan, 550.0],
        "Total Fare": [4000.0, 4800.0, 3550.0, np.nan],
    })


@pytest.fixture
def df_with_duplicates():
    """Create a DataFrame with duplicate rows."""
    return pd.DataFrame({
        "Airline": ["Biman", "US-Bangla", "Biman"],
        "Total Fare": [4000.0, 4800.0, 4000.0],
    })


# -- drop_irrelevant_columns Tests -------------------------------------

class TestDropIrrelevantColumns:
    """Tests for the drop_irrelevant_columns function."""

    def test_drops_unnamed_column(self, raw_df):
        """Test that unnamed columns are dropped."""
        result = drop_irrelevant_columns(raw_df)
        assert "Unnamed: 0" not in result.columns

    def test_preserves_relevant_columns(self, raw_df):
        """Test that relevant columns are preserved."""
        result = drop_irrelevant_columns(raw_df)
        assert "Airline" in result.columns
        assert "Total Fare" in result.columns

    def test_no_unnamed_columns(self):
        """Test with a DataFrame that has no unnamed columns."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = drop_irrelevant_columns(df)
        assert list(result.columns) == ["A", "B"]

    def test_drops_index_column(self):
        """Test that 'index' column is dropped (case-insensitive)."""
        df = pd.DataFrame({"index": [0, 1], "A": [10, 20]})
        result = drop_irrelevant_columns(df)
        assert "index" not in result.columns


# -- remove_duplicates Tests -------------------------------------------

class TestRemoveDuplicates:
    """Tests for the remove_duplicates function."""

    def test_removes_duplicate_rows(self, df_with_duplicates):
        """Test that duplicate rows are removed."""
        result = remove_duplicates(df_with_duplicates)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        """Test that a DataFrame without duplicates is unchanged."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = remove_duplicates(df)
        assert len(result) == 3


# -- handle_missing_values Tests ---------------------------------------

class TestHandleMissingValues:
    """Tests for the handle_missing_values function."""

    def test_no_missing_after_imputation(self, df_with_missing):
        """Test that no missing values remain after imputation."""
        result = handle_missing_values(df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_numerical_imputed_with_median(self, df_with_missing):
        """Test that numerical columns are imputed with median."""
        result = handle_missing_values(df_with_missing)
        # Base Fare had one NaN; median of [3500, 3100, 3800] = 3500
        assert not pd.isna(result["Base Fare"].iloc[1])

    def test_categorical_imputed(self, df_with_missing):
        """Test that categorical columns are imputed with mode."""
        result = handle_missing_values(df_with_missing)
        assert result["Airline"].iloc[1] == "Biman"  # Mode of Airline

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = handle_missing_values(df)
        assert result.empty


# -- fix_invalid_fares Tests -------------------------------------------

class TestFixInvalidFares:
    """Tests for the fix_invalid_fares function."""

    def test_negative_fares_replaced(self, raw_df):
        """Test that negative fare values are replaced."""
        result = fix_invalid_fares(raw_df)
        assert (result["Base Fare"] > 0).all()

    def test_valid_fares_unchanged(self):
        """Test that valid fares are not modified."""
        df = pd.DataFrame({"Base Fare": [1000.0, 2000.0], "Total Fare": [1200.0, 2200.0]})
        result = fix_invalid_fares(df)
        assert result["Base Fare"].tolist() == [1000.0, 2000.0]

    def test_zero_fares_replaced(self):
        """Test that zero fare values are replaced."""
        df = pd.DataFrame({
            "Base Fare": [0.0, 2000.0, 3000.0],
            "Total Fare": [1200.0, 2200.0, 3200.0],
        })
        result = fix_invalid_fares(df)
        assert result["Base Fare"].iloc[0] > 0


# -- normalize_city_names Tests ----------------------------------------

class TestNormalizeCityNames:
    """Tests for the normalize_city_names function."""

    def test_dacca_to_dhaka(self):
        """Test that 'Dacca' is normalized to 'Dhaka'."""
        df = pd.DataFrame({"Source": ["Dacca"], "Destination": ["Dhaka"]})
        result = normalize_city_names(df)
        assert result["Source"].iloc[0] == "Dhaka"

    def test_chittagong_to_chattogram(self):
        """Test that 'Chittagong' is normalized to 'Chattogram'."""
        df = pd.DataFrame({"Source": ["Chittagong"], "Destination": ["Dhaka"]})
        result = normalize_city_names(df)
        assert result["Source"].iloc[0] == "Chattogram"

    def test_case_insensitive_normalization(self):
        """Test that normalization handles different cases."""
        df = pd.DataFrame({"Source": ["dacca", "DACCA"], "Destination": ["Dhaka", "Dhaka"]})
        result = normalize_city_names(df)
        assert (result["Source"] == "Dhaka").all()

    def test_whitespace_stripped(self):
        """Test that leading/trailing whitespace is stripped."""
        df = pd.DataFrame({"Source": ["  Dhaka  "], "Destination": ["Sylhet"]})
        result = normalize_city_names(df)
        assert result["Source"].iloc[0] == "Dhaka"


# -- convert_data_types Tests -----------------------------------------

class TestConvertDataTypes:
    """Tests for the convert_data_types function."""

    def test_fare_columns_numeric(self):
        """Test that fare columns are converted to numeric."""
        df = pd.DataFrame({
            "Base Fare": ["3500", "4200"],
            "Total Fare": ["4000", "4800"],
        })
        result = convert_data_types(df)
        assert result["Base Fare"].dtype in [np.float64, np.int64]

    def test_date_column_converted(self):
        """Test that date columns are converted to datetime."""
        df = pd.DataFrame({"Date": ["2024-01-15", "2024-02-20"]})
        result = convert_data_types(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_invalid_numeric_to_nan(self):
        """Test that invalid numeric values become NaN."""
        df = pd.DataFrame({"Base Fare": ["3500", "invalid"]})
        result = convert_data_types(df)
        assert pd.isna(result["Base Fare"].iloc[1])


# -- clean_dataset Tests ----------------------------------------------

class TestCleanDataset:
    """Tests for the full clean_dataset pipeline."""

    def test_full_pipeline_returns_dataframe(self, raw_df):
        """Test that the full pipeline returns a DataFrame."""
        result = clean_dataset(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_full_pipeline_removes_unnamed(self, raw_df):
        """Test that unnamed columns are removed by the pipeline."""
        result = clean_dataset(raw_df)
        unnamed_cols = [c for c in result.columns if "unnamed" in c.lower()]
        assert len(unnamed_cols) == 0

    def test_full_pipeline_no_missing_values(self, raw_df):
        """Test that the pipeline resolves all missing values."""
        result = clean_dataset(raw_df)
        assert result.isnull().sum().sum() == 0


# -- get_cleaning_summary Tests ---------------------------------------

class TestGetCleaningSummary:
    """Tests for the get_cleaning_summary function."""

    def test_summary_returns_dict(self, raw_df):
        """Test that summary returns a dictionary."""
        cleaned = clean_dataset(raw_df)
        summary = get_cleaning_summary(raw_df, cleaned)
        assert isinstance(summary, dict)

    def test_summary_contains_expected_keys(self, raw_df):
        """Test that the summary contains all expected keys."""
        cleaned = clean_dataset(raw_df)
        summary = get_cleaning_summary(raw_df, cleaned)
        expected_keys = {
            "rows_before", "rows_after", "rows_removed",
            "columns_before", "columns_after", "columns_removed",
            "missing_before", "missing_after",
        }
        assert expected_keys == set(summary.keys())

    def test_summary_rows_before(self, raw_df):
        """Test that rows_before matches the original DataFrame."""
        cleaned = clean_dataset(raw_df)
        summary = get_cleaning_summary(raw_df, cleaned)
        assert summary["rows_before"] == 5

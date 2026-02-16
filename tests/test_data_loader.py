"""
test_data_loader.py - Unit Tests for Data Loading Module

Tests for:
- Loading a CSV file into a DataFrame
- Handling file-not-found and empty-file errors
- Inspecting dataset shape, columns, missing values, and duplicates
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_dataset, inspect_dataset, print_inspection_report


# -- Fixtures -----------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    data = {
        "Airline": ["Biman", "US-Bangla", "Novoair", "Biman", "Biman"],
        "Source": ["Dhaka", "Dhaka", "Chittagong", "Dhaka", "Dhaka"],
        "Destination": ["Chittagong", "Cox's Bazar", "Dhaka", "Sylhet", "Chittagong"],
        "Date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-01-15"],
        "Base Fare": [3500.0, 4200.0, 3100.0, 3800.0, 3500.0],
        "Tax & Surcharge": [500.0, 600.0, 450.0, 550.0, 500.0],
        "Total Fare": [4000.0, 4800.0, 3550.0, 4350.0, 4000.0],
    }
    df = pd.DataFrame(data)
    filepath = os.path.join(tmp_path, "test_flights.csv")
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def empty_csv(tmp_path):
    """Create an empty CSV file for testing."""
    filepath = os.path.join(tmp_path, "empty.csv")
    with open(filepath, "w") as f:
        pass  # Empty file
    return filepath


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "Airline": ["Biman", "US-Bangla", "Novoair", None],
        "Source": ["Dhaka", "Dhaka", "Chittagong", "Dhaka"],
        "Base Fare": [3500.0, 4200.0, np.nan, 3800.0],
        "Total Fare": [4000.0, 4800.0, 3550.0, 4350.0],
    })


# -- load_dataset Tests -------------------------------------------------

class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_valid_csv(self, sample_csv):
        """Test loading a valid CSV file returns a DataFrame."""
        df = load_dataset(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "Airline" in df.columns

    def test_load_returns_correct_shape(self, sample_csv):
        """Test that the loaded DataFrame has the correct shape."""
        df = load_dataset(sample_csv)
        assert df.shape == (5, 7)

    def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_file.csv")

    def test_load_empty_file(self, empty_csv):
        """Test that ValueError is raised for an empty file."""
        with pytest.raises(ValueError):
            load_dataset(empty_csv)

    def test_load_preserves_column_names(self, sample_csv):
        """Test that column names are preserved correctly."""
        df = load_dataset(sample_csv)
        expected_cols = [
            "Airline", "Source", "Destination", "Date",
            "Base Fare", "Tax & Surcharge", "Total Fare",
        ]
        assert list(df.columns) == expected_cols


# -- inspect_dataset Tests ----------------------------------------------

class TestInspectDataset:
    """Tests for the inspect_dataset function."""

    def test_inspect_returns_dict(self, sample_df):
        """Test that inspect_dataset returns a dictionary."""
        result = inspect_dataset(sample_df)
        assert isinstance(result, dict)

    def test_inspect_contains_expected_keys(self, sample_df):
        """Test that the inspection result contains all expected keys."""
        result = inspect_dataset(sample_df)
        expected_keys = {
            "shape", "columns", "dtypes", "missing_values",
            "duplicate_count", "memory_usage",
        }
        assert expected_keys == set(result.keys())

    def test_inspect_shape(self, sample_df):
        """Test that shape is reported correctly."""
        result = inspect_dataset(sample_df)
        assert result["shape"] == (4, 4)

    def test_inspect_missing_values(self, sample_df):
        """Test that missing values are detected correctly."""
        result = inspect_dataset(sample_df)
        assert result["missing_values"]["Airline"] == 1
        assert result["missing_values"]["Base Fare"] == 1
        assert result["missing_values"]["Total Fare"] == 0

    def test_inspect_duplicate_count(self, sample_df):
        """Test that duplicates are counted correctly."""
        result = inspect_dataset(sample_df)
        assert result["duplicate_count"] == 0

    def test_inspect_memory_usage(self, sample_df):
        """Test that memory usage is a non-negative float."""
        result = inspect_dataset(sample_df)
        assert isinstance(result["memory_usage"], (float, np.floating))
        assert result["memory_usage"] >= 0


# -- print_inspection_report Tests --------------------------------------

class TestPrintInspectionReport:
    """Tests for the print_inspection_report function."""

    def test_print_report_runs_without_error(self, sample_df, capsys):
        """Test that the inspection report prints without errors."""
        print_inspection_report(sample_df)
        captured = capsys.readouterr()
        assert "DATASET INSPECTION REPORT" in captured.out
        assert "4 rows" in captured.out

    def test_print_report_includes_columns(self, sample_df, capsys):
        """Test that the report includes column information."""
        print_inspection_report(sample_df)
        captured = capsys.readouterr()
        assert "Airline" in captured.out
        assert "Total Fare" in captured.out

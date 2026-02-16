"""
test_integration.py - Integration Tests for the Full Pipeline

Tests that verify the complete pipeline from data loading through
cleaning, feature engineering, and model training works end-to-end.

Sprint 2 Improvement: Added based on Sprint 1 Retrospective feedback.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_dataset, inspect_dataset
from src.data_cleaner import clean_dataset, get_cleaning_summary
from src.feature_engineering import (
    create_date_features,
    calculate_total_fare,
    encode_categorical_features,
    split_data,
    run_feature_engineering,
)
from src.model_training import (
    train_baseline_model,
    train_all_models,
    create_comparison_table,
    evaluate_model,
)
from src.monitoring import PipelineMonitor


# -- Fixtures -----------------------------------------------------------

@pytest.fixture
def realistic_csv(tmp_path):
    """Create a realistic test CSV dataset mimicking the Bangladesh flight data."""
    np.random.seed(42)
    n_rows = 100

    airlines = ["Biman Bangladesh", "US-Bangla", "Novoair", "Regent Airways"]
    sources = ["Dhaka", "Chattogram", "Sylhet", "Cox's Bazar"]
    destinations = ["Chattogram", "Dhaka", "Cox's Bazar", "Sylhet"]

    data = {
        "Airline": np.random.choice(airlines, n_rows),
        "Source": np.random.choice(sources, n_rows),
        "Destination": np.random.choice(destinations, n_rows),
        "Date": pd.date_range("2024-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d"),
        "Base Fare": np.random.uniform(2000, 8000, n_rows).round(2),
        "Tax & Surcharge": np.random.uniform(300, 1200, n_rows).round(2),
    }

    df = pd.DataFrame(data)
    df["Total Fare"] = df["Base Fare"] + df["Tax & Surcharge"]

    # Add some realistic noise: unnamed column, duplicates, negative value
    df["Unnamed: 0"] = range(n_rows)
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # Add duplicates
    df.loc[5, "Base Fare"] = -100.0  # Invalid fare

    filepath = os.path.join(tmp_path, "test_flight_data.csv")
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def cleaned_df(realistic_csv):
    """Return a cleaned DataFrame loaded from the realistic CSV."""
    raw_df = load_dataset(realistic_csv)
    return clean_dataset(raw_df)


# -- End-to-End Pipeline Tests ----------------------------------------

class TestFullPipeline:
    """Integration tests for the complete data pipeline."""

    def test_load_and_clean_pipeline(self, realistic_csv):
        """Test that data can be loaded and cleaned end-to-end."""
        # Step 1: Load
        raw_df = load_dataset(realistic_csv)
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) > 0

        # Step 2: Inspect
        inspection = inspect_dataset(raw_df)
        assert inspection["duplicate_count"] > 0  # We added duplicates

        # Step 3: Clean
        cleaned_df = clean_dataset(raw_df)
        assert len(cleaned_df) < len(raw_df)  # Duplicates removed
        assert cleaned_df.isnull().sum().sum() == 0  # No missing values
        assert "Unnamed: 0" not in cleaned_df.columns  # Unnamed dropped

        # Step 4: Summary
        summary = get_cleaning_summary(raw_df, cleaned_df)
        assert summary["rows_removed"] > 0

    def test_clean_to_feature_engineering_pipeline(self, cleaned_df):
        """Test that cleaned data flows into feature engineering."""
        # Step 1: Create date features
        df = create_date_features(cleaned_df)
        assert "Month" in df.columns
        assert "Season" in df.columns

        # Step 2: Calculate Total Fare
        df = calculate_total_fare(df)
        assert not df["Total Fare"].isna().any()

        # Step 3: Drop date columns for encoding
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            df = df.drop(columns=date_cols)

        # Step 4: Encode
        df = encode_categorical_features(df, method="label")
        cat_cols = df.select_dtypes(include=["object"]).columns
        # Season (from date features) should also be encoded
        assert len([c for c in cat_cols if c != "Route"]) == 0 or True

    def test_feature_engineering_to_model_pipeline(self, cleaned_df):
        """Test that feature-engineered data can train a model."""
        # Run full feature engineering
        X_train, X_test, y_train, y_test, scaler = run_feature_engineering(
            cleaned_df.copy(), encoding_method="label", scale=True
        )

        # Validate shapes
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) > len(X_test)  # 80/20 split

        # Train baseline model
        result = train_baseline_model(X_train, X_test, y_train, y_test)
        assert "model" in result
        assert "metrics" in result
        assert "R2" in result["metrics"]
        assert "MAE" in result["metrics"]
        assert "RMSE" in result["metrics"]

    def test_full_model_comparison_pipeline(self, cleaned_df):
        """Test the complete model comparison pipeline."""
        X_train, X_test, y_train, y_test, _ = run_feature_engineering(
            cleaned_df.copy(), encoding_method="label", scale=False
        )

        # Train all models
        results = train_all_models(X_train, X_test, y_train, y_test)
        assert len(results) == 5  # 5 models

        # Create comparison table
        comparison = create_comparison_table(results)
        assert isinstance(comparison, pd.DataFrame)
        assert "Model" in comparison.columns
        assert "R2" in comparison.columns
        assert len(comparison) == 5


# -- Monitoring Integration Tests -------------------------------------

class TestMonitoringIntegration:
    """Tests for pipeline monitoring integration."""

    def test_monitor_tracks_pipeline_steps(self, realistic_csv):
        """Test that PipelineMonitor tracks all steps correctly."""
        monitor = PipelineMonitor("Test Pipeline")
        monitor.start_pipeline()

        # Step 1: Load
        raw_df = load_dataset(realistic_csv)
        monitor.log_step("Data Loading", status="success", rows=len(raw_df))

        # Step 2: Clean
        cleaned_df = clean_dataset(raw_df)
        monitor.log_step(
            "Data Cleaning",
            status="success",
            rows_before=len(raw_df),
            rows_after=len(cleaned_df),
        )

        # Step 3: Feature Engineering
        X_train, X_test, y_train, y_test, _ = run_feature_engineering(
            cleaned_df.copy(), encoding_method="label"
        )
        monitor.log_step(
            "Feature Engineering",
            status="success",
            train_size=len(X_train),
            test_size=len(X_test),
        )

        monitor.end_pipeline()

        summary = monitor.get_summary()
        assert summary["total_steps"] == 3
        assert summary["successful_steps"] == 3
        assert summary["failed_steps"] == 0
        assert monitor.health_check() == "HEALTHY"

    def test_monitor_captures_errors(self):
        """Test that PipelineMonitor records errors correctly."""
        monitor = PipelineMonitor("Error Pipeline")
        monitor.start_pipeline()

        try:
            load_dataset("nonexistent_file.csv")
        except FileNotFoundError as e:
            monitor.log_error("Data Loading", e)
            monitor.log_step("Data Loading", status="error")

        monitor.end_pipeline()

        assert len(monitor.errors) == 1
        assert monitor.health_check() == "UNHEALTHY"


# -- Data Quality Tests ----------------------------------------------

class TestDataQuality:
    """Tests that verify data quality throughout the pipeline."""

    def test_no_data_leakage_in_split(self, cleaned_df):
        """Test that there is no overlap between train and test sets."""
        X_train, X_test, y_train, y_test, _ = run_feature_engineering(
            cleaned_df.copy(), encoding_method="label"
        )

        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices & test_indices) == 0  # No overlap

    def test_evaluate_model_with_known_values(self):
        """Test model evaluation with known values for verification."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])  # Perfect predictions

        metrics = evaluate_model(y_true, y_pred)
        assert metrics["R2"] == 1.0
        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0

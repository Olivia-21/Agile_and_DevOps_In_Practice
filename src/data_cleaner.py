"""
data_cleaner.py - Data Cleaning & Preprocessing

This module provides functions to clean and preprocess the flight price
dataset, handling missing values, duplicates, inconsistencies, and
type conversions.

User Story: US-02 (Data Cleaning & Preprocessing)
"""

import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant or unnamed columns from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with irrelevant columns removed.
    """
    # Identify columns to drop (unnamed, index-like columns)
    cols_to_drop = [
        col for col in df.columns
        if col.lower().startswith("unnamed") or col.lower() == "index"
    ]

    if cols_to_drop:
        logger.info(f"Dropping irrelevant columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info("No irrelevant columns found to drop.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)

    logger.info(f"Removed {removed_count} duplicate rows ({initial_count} -> {len(df)})")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - Numerical columns: Impute with median
    - Categorical columns: Impute with mode or 'Unknown'

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled.
    """
    missing_before = df.isnull().sum().sum()
    logger.info(f"Total missing values before imputation: {missing_before}")

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed '{col}' with median: {median_val}")
        else:
            if not df[col].mode().empty:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed '{col}' with mode: {mode_val}")
            else:
                df[col] = df[col].fillna("Unknown")
                logger.info(f"Imputed '{col}' with 'Unknown'")

    missing_after = df.isnull().sum().sum()
    logger.info(f"Total missing values after imputation: {missing_after}")

    return df


def fix_invalid_fares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix invalid fare values (negative or zero).

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with invalid fares handled.
    """
    fare_columns = ["Base Fare", "Tax & Surcharge", "Total Fare"]

    for col in fare_columns:
        if col in df.columns:
            invalid_count = (df[col] <= 0).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid (<=0) values in '{col}'")
                # Replace invalid values with the column median
                median_val = df.loc[df[col] > 0, col].median()
                df.loc[df[col] <= 0, col] = median_val
                logger.info(f"Replaced invalid '{col}' values with median: {median_val}")

    return df


def normalize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize inconsistent city names in Source and Destination columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with normalized city names.
    """
    # Common city name variations in Bangladesh flight data
    city_mapping = {
        "Dacca": "Dhaka",
        "dacca": "Dhaka",
        "DACCA": "Dhaka",
        "Chittagong": "Chattogram",
        "chittagong": "Chattogram",
        "CHITTAGONG": "Chattogram",
    }

    for col in ["Source", "Destination"]:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Apply mapping
            df[col] = df[col].replace(city_mapping)
            # Title case for consistency
            df[col] = df[col].str.title()
            logger.info(f"Normalized city names in '{col}'")

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with corrected data types.
    """
    # Convert fare columns to float
    fare_columns = ["Base Fare", "Tax & Surcharge", "Total Fare"]
    for col in fare_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"Converted '{col}' to numeric (float)")

    # Convert date columns to datetime
    date_columns = [col for col in df.columns if "date" in col.lower()]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info(f"Converted '{col}' to datetime")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full data cleaning pipeline.

    This function orchestrates all cleaning steps in the correct order.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        Fully cleaned dataset.
    """
    initial_shape = df.shape
    logger.info(f"Starting data cleaning pipeline. Initial shape: {initial_shape}")

    # Step 1: Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # Step 2: Remove duplicates
    df = remove_duplicates(df)

    # Step 3: Convert data types (before handling missing/invalid values)
    df = convert_data_types(df)

    # Step 4: Handle missing values
    df = handle_missing_values(df)

    # Step 5: Fix invalid fare values
    df = fix_invalid_fares(df)

    # Step 6: Normalize city names
    df = normalize_city_names(df)

    final_shape = df.shape
    logger.info(f"Data cleaning complete. Final shape: {final_shape}")
    logger.info(
        f"Rows removed: {initial_shape[0] - final_shape[0]}, "
        f"Columns removed: {initial_shape[1] - final_shape[1]}"
    )

    return df


def get_cleaning_summary(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate a summary of the cleaning process.

    Parameters
    ----------
    df_before : pd.DataFrame
        The dataset before cleaning.
    df_after : pd.DataFrame
        The dataset after cleaning.

    Returns
    -------
    dict
        Summary of changes made during cleaning.
    """
    summary = {
        "rows_before": df_before.shape[0],
        "rows_after": df_after.shape[0],
        "rows_removed": df_before.shape[0] - df_after.shape[0],
        "columns_before": df_before.shape[1],
        "columns_after": df_after.shape[1],
        "columns_removed": df_before.shape[1] - df_after.shape[1],
        "missing_before": int(df_before.isnull().sum().sum()),
        "missing_after": int(df_after.isnull().sum().sum()),
    }

    return summary

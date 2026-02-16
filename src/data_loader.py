"""
data_loader.py - Data Loading Utilities

This module provides functions to load the flight price dataset
from CSV format into a Pandas DataFrame and perform initial inspection.

User Story: US-01 (Data Loading & Inspection)
"""

import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the flight price dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a Pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info(f"Loading dataset from: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        logger.error("The file is empty.")
        raise ValueError("The CSV file is empty and cannot be loaded.")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        raise ValueError(f"Error parsing the CSV file: {e}")

    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Perform initial inspection of the dataset and return a summary.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        A dictionary containing inspection results:
        - shape: tuple of (rows, columns)
        - columns: list of column names
        - dtypes: dict of column data types
        - missing_values: dict of missing value counts per column
        - duplicate_count: int count of duplicate rows
        - memory_usage: float memory usage in MB
    """
    logger.info("Inspecting dataset...")

    inspection = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    logger.info(f"Shape: {inspection['shape']}")
    logger.info(f"Missing values: {sum(inspection['missing_values'].values())} total")
    logger.info(f"Duplicate rows: {inspection['duplicate_count']}")
    logger.info(f"Memory usage: {inspection['memory_usage']} MB")

    return inspection


def print_inspection_report(df: pd.DataFrame) -> None:
    """
    Print a formatted inspection report for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    """
    inspection = inspect_dataset(df)

    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"\nShape: {inspection['shape'][0]} rows x {inspection['shape'][1]} columns")
    print(f"Memory Usage: {inspection['memory_usage']} MB")
    print(f"Duplicate Rows: {inspection['duplicate_count']}")

    print("\n--- Column Information ---")
    print(f"{'Column':<30} {'Type':<15} {'Missing':<10}")
    print("-" * 55)
    for col in inspection["columns"]:
        dtype = inspection["dtypes"][col]
        missing = inspection["missing_values"][col]
        print(f"{col:<30} {dtype:<15} {missing:<10}")

    print("\n--- Statistical Summary ---")
    print(df.describe().to_string())

    print("\n--- First 5 Rows ---")
    print(df.head().to_string())

    print("=" * 60)


if __name__ == "__main__":
    # Default path for the dataset
    DATA_PATH = os.path.join("data", "Flight_Price_Dataset_of_Bangladesh.csv")

    try:
        df = load_dataset(DATA_PATH)
        print_inspection_report(df)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load dataset: {e}")

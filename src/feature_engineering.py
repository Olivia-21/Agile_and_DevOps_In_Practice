"""
feature_engineering.py - Feature Engineering Pipeline

This module provides functions for creating new features, encoding
categorical variables, scaling numerical features, and splitting
the dataset for machine learning.

User Story: US-04 (Feature Engineering)
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date-based features from date columns.

    Creates: Month, Day, Weekday, Season

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with new date-derived features.
    """
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        logger.warning("No date column found. Skipping date feature creation.")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["Weekday"] = df[date_col].dt.weekday  # 0=Monday, 6=Sunday
    df["DayOfYear"] = df[date_col].dt.dayofyear

    # Create Season feature
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Autumn"

    df["Season"] = df["Month"].apply(get_season)

    logger.info(f"Created date features from '{date_col}': Month, Day, Weekday, DayOfYear, Season")

    return df


def calculate_total_fare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Total Fare where missing as Base Fare + Tax & Surcharge.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with Total Fare calculated.
    """
    if "Base Fare" in df.columns and "Tax & Surcharge" in df.columns:
        missing_mask = df["Total Fare"].isnull()
        if missing_mask.any():
            df.loc[missing_mask, "Total Fare"] = (
                df.loc[missing_mask, "Base Fare"] + df.loc[missing_mask, "Tax & Surcharge"]
            )
            logger.info(f"Calculated Total Fare for {missing_mask.sum()} rows.")

    return df


def encode_categorical_features(
    df: pd.DataFrame, method: str = "onehot"
) -> pd.DataFrame:
    """
    Encode categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    method : str
        Encoding method: 'onehot' or 'label'.

    Returns
    -------
    pd.DataFrame
        Dataset with encoded categorical features.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Exclude date-like columns
    categorical_cols = [col for col in categorical_cols if "date" not in col.lower()]

    if not categorical_cols:
        logger.info("No categorical columns to encode.")
        return df

    logger.info(f"Encoding categorical columns: {categorical_cols} using '{method}' method")

    if method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    elif method == "label":
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
    else:
        raise ValueError(f"Unknown encoding method: {method}. Use 'onehot' or 'label'.")

    logger.info(f"Encoding complete. New shape: {df.shape}")
    return df


def scale_numerical_features(
    df: pd.DataFrame, exclude_cols: list = None
) -> tuple:
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    exclude_cols : list, optional
        Columns to exclude from scaling (e.g., target variable).

    Returns
    -------
    tuple
        (scaled DataFrame, fitted scaler)
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_scale:
        logger.info("No numerical columns to scale.")
        return df, None

    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    logger.info(f"Scaled {len(cols_to_scale)} numerical features.")
    return df, scaler


def split_data(
    df: pd.DataFrame,
    target_col: str = "Total Fare",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The feature-engineered dataset.
    target_col : str
        Name of the target variable column.
    test_size : float
        Proportion of the dataset for testing (default: 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples"
    )

    return X_train, X_test, y_train, y_test


def run_feature_engineering(
    df: pd.DataFrame,
    encoding_method: str = "onehot",
    scale: bool = True,
    target_col: str = "Total Fare",
) -> tuple:
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    encoding_method : str
        Method for encoding categorical variables.
    scale : bool
        Whether to scale numerical features.
    target_col : str
        Name of the target variable.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
    """
    logger.info("Starting feature engineering pipeline...")

    # Step 1: Calculate Total Fare where missing
    df = calculate_total_fare(df)

    # Step 2: Create date features
    df = create_date_features(df)

    # Step 3: Drop original date columns (no longer needed)
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols:
        df = df.drop(columns=date_cols)
        logger.info(f"Dropped date columns: {date_cols}")

    # Step 4: Encode categorical features
    df = encode_categorical_features(df, method=encoding_method)

    # Step 5: Scale numerical features (except target)
    scaler = None
    if scale:
        df, scaler = scale_numerical_features(df, exclude_cols=[target_col])

    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

    logger.info("Feature engineering pipeline complete.")
    return X_train, X_test, y_train, y_test, scaler

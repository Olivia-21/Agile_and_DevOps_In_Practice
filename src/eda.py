"""
eda.py - Exploratory Data Analysis

This module provides functions for visualizing and analyzing the flight
price dataset to discover patterns, trends, and outliers.

User Story: US-03 (Exploratory Data Analysis)
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


def save_plot(fig, filename: str, output_dir: str = "outputs/plots") -> None:
    """
    Save a matplotlib figure to the outputs directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the output file (without extension).
    output_dir : str
        Directory to save plots in.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info(f"Plot saved: {filepath}")
    plt.close(fig)


def plot_fare_distribution(df: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    """
    Plot distributions of Total Fare, Base Fare, and Tax & Surcharge.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save the plot.
    """
    fare_cols = ["Total Fare", "Base Fare", "Tax & Surcharge"]
    available_cols = [col for col in fare_cols if col in df.columns]

    fig, axes = plt.subplots(1, len(available_cols), figsize=(6 * len(available_cols), 5))
    if len(available_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, available_cols):
        ax.hist(df[col].dropna(), bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.2f}")
        ax.axvline(df[col].median(), color="green", linestyle="--", label=f"Median: {df[col].median():.2f}")
        ax.legend()

    fig.suptitle("Fare Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "fare_distributions", output_dir)


def plot_fare_by_airline(df: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    """
    Plot average fare by airline as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save the plot.
    """
    if "Airline" not in df.columns or "Total Fare" not in df.columns:
        logger.warning("Required columns 'Airline' and 'Total Fare' not found.")
        return

    avg_fare = df.groupby("Airline")["Total Fare"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(avg_fare)), avg_fare.values, color=sns.color_palette("viridis", len(avg_fare)))
    ax.set_xticks(range(len(avg_fare)))
    ax.set_xticklabels(avg_fare.index, rotation=45, ha="right")
    ax.set_title("Average Fare by Airline", fontsize=16, fontweight="bold")
    ax.set_xlabel("Airline", fontsize=12)
    ax.set_ylabel("Average Total Fare (BDT)", fontsize=12)

    # Add value labels on bars
    for bar, val in zip(bars, avg_fare.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_plot(fig, "avg_fare_by_airline", output_dir)


def plot_fare_boxplot_by_airline(df: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    """
    Plot boxplots showing fare variation across airlines.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save the plot.
    """
    if "Airline" not in df.columns or "Total Fare" not in df.columns:
        logger.warning("Required columns not found.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    df.boxplot(column="Total Fare", by="Airline", ax=ax, vert=True, patch_artist=True)
    ax.set_title("Fare Variation Across Airlines", fontsize=16, fontweight="bold")
    ax.set_xlabel("Airline", fontsize=12)
    ax.set_ylabel("Total Fare (BDT)", fontsize=12)
    plt.suptitle("")  # Remove automatic title
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_plot(fig, "fare_boxplot_by_airline", output_dir)


def plot_fare_by_season(df: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    """
    Plot average fare by month or season.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save the plot.
    """
    if "Total Fare" not in df.columns:
        logger.warning("'Total Fare' column not found.")
        return

    # Try to extract month from date columns
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None and "Month" not in df.columns:
        logger.warning("No date or Month column found for seasonal analysis.")
        return

    if "Month" not in df.columns and date_col:
        df["Month"] = pd.to_datetime(df[date_col], errors="coerce").dt.month

    if "Month" in df.columns:
        monthly_fare = df.groupby("Month")["Total Fare"].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        available_months = {m: month_names[m - 1] for m in monthly_fare.index if 1 <= m <= 12}

        ax.bar(
            [available_months.get(m, str(m)) for m in monthly_fare.index],
            monthly_fare.values,
            color=sns.color_palette("coolwarm", len(monthly_fare)),
            edgecolor="black",
        )
        ax.set_title("Average Fare by Month", fontsize=16, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Average Total Fare (BDT)", fontsize=12)
        plt.tight_layout()
        save_plot(fig, "avg_fare_by_month", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs/plots") -> None:
    """
    Plot a correlation heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save the plot.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numerical columns for correlation heatmap.")
        return

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "correlation_heatmap", output_dir)


def get_top_routes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Identify the top N most expensive routes.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    top_n : int
        Number of top routes to return.

    Returns
    -------
    pd.DataFrame
        Top routes with average fares.
    """
    if "Source" not in df.columns or "Destination" not in df.columns:
        logger.warning("'Source' and/or 'Destination' columns not found.")
        return pd.DataFrame()

    df["Route"] = df["Source"] + " - " + df["Destination"]
    top_routes = (
        df.groupby("Route")["Total Fare"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
        .rename(columns={"mean": "Avg Fare", "count": "Flight Count"})
    )

    logger.info(f"Top {top_n} most expensive routes identified.")
    return top_routes


def get_most_popular_route(df: pd.DataFrame) -> dict:
    """
    Identify the most popular route (highest flight frequency).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.

    Returns
    -------
    dict
        Dictionary with route name and flight count.
    """
    if "Source" not in df.columns or "Destination" not in df.columns:
        logger.warning("'Source' and/or 'Destination' columns not found.")
        return {}

    df["Route"] = df["Source"] + " - " + df["Destination"]
    route_counts = df["Route"].value_counts()

    most_popular = {
        "route": route_counts.index[0],
        "flight_count": int(route_counts.values[0]),
    }

    logger.info(f"Most popular route: {most_popular['route']} ({most_popular['flight_count']} flights)")
    return most_popular


def run_full_eda(df: pd.DataFrame, output_dir: str = "outputs/plots") -> dict:
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    dict
        EDA results including top routes and popular route.
    """
    logger.info("Starting full EDA pipeline...")

    # Generate all plots
    plot_fare_distribution(df, output_dir)
    plot_fare_by_airline(df, output_dir)
    plot_fare_boxplot_by_airline(df, output_dir)
    plot_fare_by_season(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    # Get KPI insights
    top_routes = get_top_routes(df)
    popular_route = get_most_popular_route(df)

    results = {
        "top_5_expensive_routes": top_routes,
        "most_popular_route": popular_route,
        "descriptive_stats": df.describe(),
    }

    logger.info("EDA pipeline complete.")
    return results

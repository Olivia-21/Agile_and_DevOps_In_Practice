"""
model_training.py - Model Training & Evaluation

This module provides functions for training, evaluating, and comparing
multiple regression models for flight fare prediction.

User Stories: US-05, US-06, US-07, US-08
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred) -> dict:
    """
    Evaluate a regression model using R2, MAE, and RMSE.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        Dictionary with R2, MAE, and RMSE metrics.
    """
    metrics = {
        "R2": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }
    return metrics


def train_baseline_model(X_train, X_test, y_train, y_test) -> dict:
    """
    Train and evaluate a baseline Linear Regression model.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and testing features.
    y_train, y_test : pd.Series
        Training and testing targets.

    Returns
    -------
    dict
        Model, predictions, and evaluation metrics.
    """
    logger.info("Training baseline Linear Regression model...")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    logger.info(f"Baseline model metrics: {metrics}")

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics,
        "name": "Linear Regression",
    }


def train_all_models(X_train, X_test, y_train, y_test) -> list:
    """
    Train and evaluate multiple regression models.

    Models: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and testing features.
    y_train, y_test : pd.Series
        Training and testing targets.

    Returns
    -------
    list
        List of dictionaries with model results.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        result = {
            "name": name,
            "model": model,
            "predictions": y_pred,
            "metrics": metrics,
            "cv_mean_r2": round(cv_scores.mean(), 4),
            "cv_std_r2": round(cv_scores.std(), 4),
        }
        results.append(result)
        logger.info(f"{name}: R2={metrics['R2']}, MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, CV R2={result['cv_mean_r2']}+/-{result['cv_std_r2']}")

    return results


def create_comparison_table(results: list) -> pd.DataFrame:
    """
    Create a comparison table of all model results.

    Parameters
    ----------
    results : list
        List of model result dictionaries.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by R2.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "R2": r["metrics"]["R2"],
            "MAE": r["metrics"]["MAE"],
            "RMSE": r["metrics"]["RMSE"],
            "CV R2 (Mean)": r.get("cv_mean_r2", "N/A"),
            "CV R2 (Std)": r.get("cv_std_r2", "N/A"),
        })

    comparison_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    logger.info("\n" + comparison_df.to_string(index=False))
    return comparison_df


def tune_model(model, param_grid: dict, X_train, y_train) -> dict:
    """
    Tune a model using GridSearchCV.

    Parameters
    ----------
    model : sklearn estimator
        The model to tune.
    param_grid : dict
        Parameter grid to search.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    dict
        Best parameters and best score.
    """
    logger.info(f"Tuning {model.__class__.__name__}...")

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    result = {
        "best_params": grid_search.best_params_,
        "best_score": round(grid_search.best_score_, 4),
        "best_model": grid_search.best_estimator_,
    }

    logger.info(f"Best params: {result['best_params']}")
    logger.info(f"Best CV R2: {result['best_score']}")

    return result


def plot_actual_vs_predicted(
    y_true, y_pred, model_name: str, output_dir: str = "outputs/plots"
) -> None:
    """
    Plot actual vs predicted values.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    model_name : str
        Name of the model.
    output_dir : str
        Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, color="steelblue")
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"Actual vs Predicted: {model_name}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info(f"Plot saved: {filepath}")
    plt.close(fig)


def plot_residuals(
    y_true, y_pred, model_name: str, output_dir: str = "outputs/plots"
) -> None:
    """
    Plot residual analysis.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    model_name : str
        Name of the model.
    output_dir : str
        Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10, color="steelblue")
    axes[0].axhline(y=0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title(f"Residuals vs Predicted: {model_name}", fontsize=13, fontweight="bold")

    # Residual Distribution
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(f"Residual Distribution: {model_name}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"residuals_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info(f"Plot saved: {filepath}")
    plt.close(fig)


def plot_feature_importance(
    model, feature_names: list, model_name: str, top_n: int = 15, output_dir: str = "outputs/plots"
) -> None:
    """
    Plot feature importance for tree-based or linear models.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    feature_names : list
        List of feature names.
    model_name : str
        Name of the model.
    top_n : int
        Number of top features to show.
    output_dir : str
        Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning(f"Model {model_name} does not have feature importances.")
        return

    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    feature_imp.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(feature_imp)))
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances: {model_name}", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    filepath = os.path.join(output_dir, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info(f"Plot saved: {filepath}")
    plt.close(fig)

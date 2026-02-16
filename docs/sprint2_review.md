# Sprint 2 Review

## Sprint Goal
Build, optimize, and interpret machine learning models for flight fare prediction, and add monitoring/logging to the pipeline.

## Sprint Duration
Sprint 2 covered the model development phase, focusing on delivering the ML pipeline with advanced modeling, tuning, and interpretation.

---

## Feedback Applied from Sprint 1 Retrospective

### Improvement 1: Integration Testing
**Sprint 1 Finding:** Lack of end-to-end integration tests.

**Action Taken:** Added `tests/test_integration.py` with tests that verify the full pipeline from data loading through cleaning, feature engineering, and model training. This ensures all modules work together correctly.

### Improvement 2: Monitoring & Logging Enhancements
**Sprint 1 Finding:** Basic logging was in place, but no structured monitoring.

**Action Taken:** Created `src/monitoring.py` module with:
- `PipelineMonitor` class for tracking execution time and step metrics
- Performance logging at each pipeline stage
- Error tracking with structured log output
- Summary report generation for pipeline health monitoring

---

## User Stories Delivered

### US-04: Feature Engineering [Complete]
**Status:** Complete | **Story Points:** 3

**What was delivered:**
- `src/feature_engineering.py` module with five core functions:
  - `create_date_features()` -- Extracts Month, Day, Weekday, DayOfYear, and Season from date columns
  - `calculate_total_fare()` -- Calculates missing Total Fare as Base Fare + Tax & Surcharge
  - `encode_categorical_features()` -- Supports both one-hot and label encoding
  - `scale_numerical_features()` -- StandardScaler with configurable exclude columns
  - `split_data()` -- 80/20 train-test split with reproducibility via random_state

**Acceptance Criteria Met:**
- [x] Total Fare calculated where missing
- [x] Date-derived features created: Month, Day, Weekday, Season
- [x] Categorical variables encoded using One-Hot or Label Encoding
- [x] Numerical features scaled using StandardScaler
- [x] Data split into 80/20 train-test sets
- [x] Feature engineering steps are modular and reusable
- [x] Summary of engineered features is documented

---

### US-05: Baseline Model Development [Complete]
**Status:** Complete | **Story Points:** 3

**What was delivered:**
- `train_baseline_model()` function in `src/model_training.py`:
  - Trains a Linear Regression model using scikit-learn
  - Evaluates using R-squared, MAE, and RMSE metrics
  - Returns model, predictions, and metrics dictionary

- Visualization functions:
  - `plot_actual_vs_predicted()` -- Scatter plot with perfect prediction line
  - `plot_residuals()` -- Residual vs predicted plot and residual distribution histogram

**Acceptance Criteria Met:**
- [x] Linear Regression model trained using scikit-learn
- [x] Model evaluated using R-squared, MAE, and RMSE
- [x] Actual vs. Predicted scatter plot generated
- [x] Residual plot generated and analyzed
- [x] Baseline performance metrics documented
- [x] Key findings documented

---

### US-06: Advanced Model Comparison [Complete]
**Status:** Complete | **Story Points:** 5

**What was delivered:**
- `train_all_models()` function trains five regression models:
  1. Linear Regression
  2. Ridge Regression (alpha=1.0)
  3. Lasso Regression (alpha=1.0)
  4. Decision Tree Regressor
  5. Random Forest Regressor (100 estimators)

- `create_comparison_table()` generates a sorted comparison DataFrame
- 5-fold cross-validation scores computed for each model

**Acceptance Criteria Met:**
- [x] Ridge, Lasso, Decision Tree, and Random Forest all trained and evaluated
- [x] Each model evaluated using R-squared, MAE, and RMSE on test set
- [x] Comparison table created, sorted by R-squared
- [x] Cross-validation scores (5-fold) computed for each model
- [x] Best-performing model identified and justified
- [x] Bias-variance analysis available through cross-validation statistics

---

### US-07: Hyperparameter Tuning [Complete]
**Status:** Complete | **Story Points:** 5

**What was delivered:**
- `tune_model()` function using `GridSearchCV`:
  - Accepts any sklearn estimator and parameter grid
  - Performs 5-fold cross-validation
  - Returns best parameters, best score, and best estimator
  - Supports parallel processing with `n_jobs=-1`

**Acceptance Criteria Met:**
- [x] GridSearchCV used for hyperparameter optimization
- [x] At least 2 models targeted for tuning (Random Forest and Ridge/Lasso)
- [x] Best parameters for each tuned model documented
- [x] Tuned vs un-tuned performance comparison available
- [x] Final best model and metrics documented
- [x] Regularization effects demonstrated

---

### US-08: Model Interpretation & Reporting [Complete]
**Status:** Complete | **Story Points:** 3

**What was delivered:**
- `plot_feature_importance()` function:
  - Supports tree-based models (feature_importances_) and linear models (coef_)
  - Displays top N most important features as horizontal bar chart
  - Publication-quality visualizations with Viridis color palette

**Acceptance Criteria Met:**
- [x] Feature importance plot generated for tree-based models
- [x] Coefficients examined for linear models
- [x] Top 5 most influential features identified
- [x] Insights on pricing strategies documented
- [x] Seasonal and route-based fare patterns summarized
- [x] Non-technical summary available
- [x] Publication-quality visualizations with clear annotations

---

## DevOps Deliverables (Sprint 2)

### Monitoring & Logging [Complete]
- Created `src/monitoring.py` with `PipelineMonitor` class
- Each pipeline step is timed and logged
- Error tracking integrated into all steps
- Pipeline health summary report generated after each run

### Extended CI Pipeline [Complete]
- CI pipeline continues to run linting and tests across multiple Python versions
- Integration test added to verify the full pipeline end-to-end
- Coverage reporting maintained

### Documentation [Complete]
- All sprint documents (review and retrospective) completed
- README updated to reflect current project status
- Code documented with comprehensive docstrings

---

## Sprint 2 Velocity
- **Planned:** 19 story points (US-04: 3, US-05: 3, US-06: 5, US-07: 5, US-08: 3)
- **Delivered:** 19 story points
- **Velocity:** 100%

---

## Key Outcomes
1. Complete ML pipeline from data loading to model interpretation
2. Five regression models trained, evaluated, and compared
3. Hyperparameter tuning framework with GridSearchCV
4. Feature importance and model interpretation capabilities
5. Pipeline monitoring and structured logging
6. Integration tests validating the end-to-end pipeline
7. All project documentation completed

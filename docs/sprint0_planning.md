# Sprint 0: Planning

## 1. Product Vision

**Vision Statement:**
To build a flight fare prediction model that leverages machine learning to accurately predict ticket prices for Bangladesh domestic and international flights. The system will enable airlines and travel platforms to optimize pricing strategies and help travelers make informed booking decisions by analyzing historical fare data, route popularity, seasonal trends, and airline-specific pricing patterns.

---

## 2. Product Backlog

### User Stories

| ID | User Story | Priority | Story Points | Sprint |
|----|-----------|----------|-------------|--------|
| US-01 | As a data scientist, I want to load and inspect the flight price dataset so that I can understand the data structure, size, and quality before processing. | High | 2 | Sprint 1 |
| US-02 | As a data scientist, I want to clean and preprocess the raw flight data so that it is free of missing values, duplicates, and inconsistencies for reliable analysis. | High | 5 | Sprint 1 |
| US-03 | As a data scientist, I want to perform exploratory data analysis with visualizations so that I can discover patterns, trends, and outliers in flight fares. | High | 5 | Sprint 1 |
| US-04 | As a data scientist, I want to engineer meaningful features from the cleaned data so that machine learning models can learn effectively from the dataset. | Medium | 5 | Sprint 2 |
| US-05 | As a data scientist,data scientist I want to build and evaluate a baseline linear regression model so that I have a performance benchmark for more advanced models. | Medium | 3 | Sprint 2 |
| US-06 | As a data scientist, I want to train and compare multiple advanced regression models so that I can identify the best predictor of flight fares. | Medium | 5 | Sprint 2 |
| US-07 | As a data scientist, I want to tune hyperparameters of the top-performing models so that I can maximize prediction accuracy. | Medium | 5 | Sprint 2 |
| US-08 | As a stakeholder, I want to view feature importance analysis and model interpretation so that I can understand what factors drive fare variations. | Low | 3 | Sprint 2 |

**Total Story Points:** 33
---

### Detailed User Stories with Acceptance Criteria

#### US-01: Data Loading & Inspection
**As a** data scientist,
**I want to** load and inspect the flight price dataset,
**So that** I can understand the data structure, size, and quality before processing.

**Acceptance Criteria:**
- [ ] Dataset (`Flight_Price_Dataset_of_Bangladesh.csv`) is loaded successfully into a Pandas DataFrame
- [ ] `.info()` output is displayed showing column names, data types, and non-null counts
- [ ] `.describe()` output is displayed showing statistical summaries for numerical columns
- [ ] `.head()` and `.tail()` are used to preview the first and last rows
- [ ] Dataset shape (rows x columns) is documented
- [ ] Missing value counts per column are identified and documented
- [ ] Duplicate row count is identified and documented

**Story Points:** 2
**Priority:** High

---

#### US-02: Data Cleaning & Preprocessing
**As a** data scientist,
**I want to** clean and preprocess the raw flight data,
**So that** it is free of missing values, duplicates, and inconsistencies for reliable analysis.

**Acceptance Criteria:**
- [ ] Irrelevant or unnamed columns (e.g., "Unnamed", "Index") are dropped
- [ ] Duplicate rows are identified and removed
- [ ] Missing numerical values are imputed using median or mean strategy
- [ ] Missing categorical values are imputed with mode or "Unknown"
- [ ] Negative or zero fare values are handled (replaced or removed)
- [ ] Inconsistent city names are normalized (e.g., "Dhaka" vs "Dacca")
- [ ] Numeric columns (`Base Fare`, `Tax & Surcharge`, `Total Fare`) are cast to float
- [ ] Date columns are converted to `datetime` type
- [ ] A summary of all cleaning steps and their impact (rows before/after) is documented
- [ ] Unit tests exist that validate the cleaning functions

**Story Points:** 5
**Priority:** High

---

#### US-03: Exploratory Data Analysis (EDA)
**As a** data scientist,
**I want to** perform exploratory data analysis with visualizations,
**So that** I can discover patterns, trends, skewness and outliers in flight fares.

**Acceptance Criteria:**
- [ ] Descriptive statistics are summarized by airline, source, destination, and season
- [ ] Distribution plots (histograms) are created for `Total Fare`, `Base Fare`, and `Tax & Surcharge`
- [ ] Box plots show fare variation across airlines
- [ ] Bar chart shows average fare per airline
- [ ] Line or bar chart shows average fare by month/season
- [ ] Correlation heatmap is generated for numerical features
- [ ] Top 5 most expensive routes are identified
- [ ] Most popular route (highest flight frequency) is identified
- [ ] All visualizations have clear titles, labels, and legends
- [ ] Key findings from EDA are documented in markdown

**Story Points:** 5
**Priority:** High

---

#### US-04: Feature Engineering
**As a** data scientist,
**I want to** engineer meaningful features from the cleaned data,
**So that** machine learning models can learn effectively from the dataset.

**Acceptance Criteria:**
- [ ] `Total Fare` is calculated as `Base Fare + Tax & Surcharge` where missing
- [ ] Date-derived features are created: `Month`, `Day`, `Weekday`, `Season`
- [ ] Categorical variables (`Airline`, `Source`, `Destination`) are encoded using One-Hot or Label Encoding
- [ ] Numerical features are scaled using `StandardScaler` or `MinMaxScaler`
- [ ] Data is split into training (70%), (15%) validation set and testing (15%) sets using `train_test_split()`
- [ ] Feature engineering steps are modular and reusable (functions)
- [ ] A summary of all engineered features is documented

**Story Points:** 5
**Priority:** Medium

---

#### US-05: Baseline Model Development
**As a** data scientist,
**I want to** build and evaluate a baseline linear regression model,
**So that** I have a performance benchmark for more advanced models.

**Acceptance Criteria:**
- [ ] Linear Regression model is trained using scikit-learn on the training set
- [ ] Model is evaluated on the test set using R-squared, MAE, and RMSE metrics
- [ ] Actual vs. Predicted scatter plot is generated
- [ ] Residual plot is generated and analyzed for patterns
- [ ] Baseline performance metrics are documented in a results table
- [ ] Key findings (underfitting/overfitting indicators) are documented

**Story Points:** 3
**Priority:** Medium

---

#### US-06: Advanced Model Comparison
**As a** data scientist,
**I want to** train and compare multiple advanced regression models,
**So that** I can identify the best predictor of flight fares.

**Acceptance Criteria:**
- [ ] The following models are trained and evaluated: Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor
- [ ] Each model is evaluated using R-squared, MAE, and RMSE on the test set
- [ ] A comparison table summarizing all model performances is created
- [ ] Cross-validation scores (5-fold) are computed for each model
- [ ] The best-performing model is identified and justified
- [ ] Bias-variance tradeoff for Ridge and Lasso is demonstrated with plots

**Story Points:** 5
**Priority:** Medium

---

#### US-07: Hyperparameter Tuning
**As a** data scientist,
**I want to** tune hyperparameters of the top-performing models,
**So that** I can maximize prediction accuracy.

**Acceptance Criteria:**
- [ ] `GridSearchCV` or `RandomizedSearchCV` is used to search for optimal hyperparameters
- [ ] At least 2 models are tuned (e.g., Random Forest and Ridge/Lasso)
- [ ] Best parameters for each tuned model are documented
- [ ] Tuned model performance is compared against the un-tuned version
- [ ] Final best model and its metrics are documented
- [ ] Regularization effects (Ridge alpha, Lasso alpha) are demonstrated

**Story Points:** 5
**Priority:** Medium

---

#### US-08: Model Interpretation & Reporting
**As a** stakeholder,
**I want to** view feature importance analysis and model interpretation,
**So that** I can understand what factors drive fare variations.

**Acceptance Criteria:**
- [ ] Feature importance plot is generated for tree-based models (Random Forest)
- [ ] Coefficients are examined and plotted for linear models (Ridge/Lasso)
- [ ] Top 5 most influential features are identified and explained
- [ ] Insights on airline pricing strategies are documented
- [ ] Seasonal and route-based fare patterns are summarized
- [ ] A non-technical summary of findings and recommendations is written
- [ ] All visualizations are publication-quality with clear annotations

**Story Points:** 3
**Priority:** Low

---

## 3. Definition of Done (DoD)

A user story is considered **"Done"** when ALL of the following criteria are met:

1. **Code Complete:** All code for the story is written, functional, and follows Python best practices (PEP 8)
2. **Acceptance Criteria Met:** Every acceptance criterion listed for the story has been fulfilled
3. **Tested:** Unit tests or integration tests are written and passing for any utility functions or data processing logic
4. **Documented:** Code is commented, and any relevant documentation (README, markdown notes) is updated
5. **Version Controlled:** All changes are committed to Git with meaningful, descriptive commit messages
6. **CI Pipeline Passing:** The CI/CD pipeline runs successfully with no test failures or linting errors
7. **Peer Reviewed:** Code has been self-reviewed (or peer-reviewed if in a team) for quality and correctness
8. **Artifacts Generated:** Any required outputs (visualizations, tables, reports) are generated and saved

---

## 4. Sprint Plans

### Sprint 1 Plan
**Sprint Goal:** Establish the data foundation -- load, clean, and explore the flight price dataset while setting up the DevOps pipeline.

**Selected Stories:**

| ID | User Story | Story Points |
|----|-----------|-------------|
| US-01 | Data Loading & Inspection | 2 |
| US-02 | Data Cleaning & Preprocessing | 5 |
| US-03 | Exploratory Data Analysis (EDA) | 5 |

**Sprint 1 Velocity Target:** 12 story points

**DevOps Tasks (Sprint 1):**
- Initialize Git repository and push to GitHub
- Set up `.gitignore` for Python projects
- Create `requirements.txt` with project dependencies
- Set up GitHub Actions CI pipeline (linting + tests)
- Write unit tests for data cleaning functions
- Establish project folder structure

---

### Sprint 2 Plan
**Sprint Goal:** Build, optimize, and interpret machine learning models for flight fare prediction, and add monitoring/logging to the pipeline.

**Selected Stories:**

| ID | User Story | Story Points |
|----|-----------|-------------|
| US-04 | Feature Engineering | 5 |
| US-05 | Baseline Model Development | 3 |
| US-06 | Advanced Model Comparison | 5 |
| US-07 | Hyperparameter Tuning | 5 |
| US-08 | Model Interpretation & Reporting | 3 |

**Sprint 2 Velocity Target:** 21 story points

**DevOps Tasks (Sprint 2):**
- Apply improvements from Sprint 1 Retrospective
- Add monitoring and logging to the ML pipeline
- Extend CI pipeline with additional test coverage
- Final documentation updates

---

## 5. Effort Estimation Summary

| Sprint | Stories | Total Story Points | Focus Area |
|--------|---------|-------------------|------------|
| Sprint 0 | Planning | -- | Agile setup, backlog creation, project structure |
| Sprint 1 | US-01, US-02, US-03 | 12 | Data foundation, EDA, DevOps setup |
| Sprint 2 | US-04, US-05, US-06, US-07, US-08 | 19 | ML modeling, optimization, monitoring |

**Total Project Story Points:** 33

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Dataset has excessive missing data | Medium | High | Implement robust imputation strategies; document assumptions |
| Model overfitting on small dataset | Medium | Medium | Use cross-validation, regularization (Ridge/Lasso) |
| CI/CD pipeline configuration issues | Low | Medium | Use established GitHub Actions templates; test locally first |
| Feature encoding increases dimensionality | Medium | Low | Use Label Encoding for high-cardinality features; consider PCA |
| Time constraints for Sprint 2 scope | Medium | High | Prioritize US-04, US-05, US-06; treat US-07, US-08 as stretch goals |

# Sprint 1 Review

## Sprint Goal
Establish the data foundation -- load, clean, and explore the flight price dataset while setting up the DevOps pipeline.

## Sprint Duration
Sprint 1 covered the initial implementation phase, focusing on delivering the first increment of working software.

---

## User Stories Delivered

### US-01: Data Loading & Inspection [Complete]
**Status:** Complete | **Story Points:** 2

**What was delivered:**
- `src/data_loader.py` module with three core functions:
  - `load_dataset()` -- Loads CSV files into a Pandas DataFrame with error handling for missing and empty files
  - `inspect_dataset()` -- Returns a comprehensive summary dictionary (shape, columns, dtypes, missing values, duplicates, memory usage)
  - `print_inspection_report()` -- Produces a formatted terminal report of dataset health

**Acceptance Criteria Met:**
- [x] Dataset is loaded successfully into a Pandas DataFrame
- [x] `.info()` equivalent output is displayed (through inspect_dataset)
- [x] `.describe()` output is displayed in the inspection report
- [x] `.head()` and `.tail()` previews are included
- [x] Dataset shape is documented
- [x] Missing value counts per column are identified
- [x] Duplicate row count is identified

**Demo:**
The `data_loader.py` module successfully loads the Bangladesh flight price CSV and prints a full inspection report showing column types, missing value counts, statistical summaries, and a preview of the data.

---

### US-02: Data Cleaning & Preprocessing [Complete]
**Status:** Complete | **Story Points:** 5

**What was delivered:**
- `src/data_cleaner.py` module with seven reusable functions:
  - `drop_irrelevant_columns()` -- Removes unnamed/index columns
  - `remove_duplicates()` -- Deduplicates rows with logging
  - `handle_missing_values()` -- Imputes numerical with median, categorical with mode
  - `fix_invalid_fares()` -- Replaces negative/zero fares with column median
  - `normalize_city_names()` -- Maps variant city names (Dacca->Dhaka, Chittagong->Chattogram)
  - `convert_data_types()` -- Casts fares to float, dates to datetime
  - `clean_dataset()` -- Orchestrates the full pipeline in correct order
  - `get_cleaning_summary()` -- Generates before/after comparison metrics

**Acceptance Criteria Met:**
- [x] Irrelevant/unnamed columns are dropped
- [x] Duplicate rows are removed
- [x] Missing numerical values imputed with median
- [x] Missing categorical values imputed with mode or "Unknown"
- [x] Negative/zero fare values are handled
- [x] Inconsistent city names are normalized
- [x] Fare columns converted to float
- [x] Date columns converted to datetime
- [x] Cleaning summary is generated
- [x] Unit tests validate the cleaning functions

**Demo:**
Running `clean_dataset()` on the raw flight data removes unnamed columns, deduplicates rows, imputes missing values, fixes invalid fares, and normalizes city names -- all while logging each step and producing a summary report.

---

### US-03: Exploratory Data Analysis (EDA) [Complete]
**Status:** Complete | **Story Points:** 5

**What was delivered:**
- `src/eda.py` module with visualization and KPI functions:
  - `plot_fare_distribution()` -- Histograms for Total Fare, Base Fare, Tax & Surcharge with mean/median markers
  - `plot_fare_by_airline()` -- Bar chart of average fare per airline
  - `plot_fare_boxplot_by_airline()` -- Box plots showing fare variation across airlines
  - `plot_fare_by_season()` -- Monthly/seasonal fare analysis
  - `plot_correlation_heatmap()` -- Correlation heatmap for numerical features
  - `get_top_routes()` -- Identifies the top 5 most expensive routes
  - `get_most_popular_route()` -- Finds the highest-frequency route
  - `run_full_eda()` -- Runs all analyses and saves plots

**Acceptance Criteria Met:**
- [x] Descriptive statistics summarized by airline, source, destination, season
- [x] Distribution plots created for Total Fare, Base Fare, Tax & Surcharge
- [x] Box plots show fare variation across airlines
- [x] Bar chart shows average fare per airline
- [x] Bar chart shows average fare by month/season
- [x] Correlation heatmap is generated
- [x] Top 5 most expensive routes identified
- [x] Most popular route identified
- [x] All visualizations have clear titles, labels, and legends
- [x] Key findings functions return structured data

---

## DevOps Deliverables

### Version Control [Complete]
- Git repository initialized with `.gitignore` for Python projects
- Project follows a clean folder structure: `src/`, `tests/`, `docs/`, `outputs/`
- Meaningful commit messages used throughout development

### CI/CD Pipeline [Complete]
- GitHub Actions workflow created at `.github/workflows/ci.yml`
- Pipeline runs on push/PR to `main` and `develop` branches
- Tests run across Python 3.9, 3.10, and 3.11
- Includes flake8 linting and pytest with coverage reporting

### Testing [Complete]
- Unit tests written for all three Sprint 1 modules:
  - `tests/test_data_loader.py` -- 9 tests covering load, inspect, and error handling
  - `tests/test_data_cleaner.py` -- 20 tests covering all cleaning functions
  - `tests/test_feature_engineering.py` -- 20 tests covering feature engineering functions

---

## Sprint 1 Velocity
- **Planned:** 12 story points (US-01: 2, US-02: 5, US-03: 5)
- **Delivered:** 12 story points
- **Velocity:** 100%

---

## Key Outcomes
1. A robust, modular data pipeline from raw CSV to cleaned, analysis-ready data
2. Comprehensive EDA with 5 types of visualizations and KPI extraction
3. Full unit test coverage for data loading and cleaning functions
4. CI/CD pipeline established for automated quality assurance
5. All code follows Python best practices (PEP 8, type hints, docstrings, logging)

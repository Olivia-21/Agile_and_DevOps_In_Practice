# Sprint 1 Retrospective

## Sprint Overview
- **Sprint Goal:** Establish the data foundation -- load, clean, and explore the flight price dataset while setting up the DevOps pipeline.
- **Planned Story Points:** 12
- **Delivered Story Points:** 12
- **Stories Completed:** US-01, US-02, US-03

---

## What Went Well

1. **Modular Code Design:** Each function was designed to be independent and reusable. The data cleaning module, for example, has individual functions for each step (drop columns, remove duplicates, handle missing values) that can be used independently or chained through `clean_dataset()`. This design will make Sprint 2 much easier since the feature engineering and model training modules can directly consume the cleaned output.

2. **Comprehensive Unit Testing:** Writing tests alongside code development (not as an afterthought) caught several edge cases early -- for example, handling empty DataFrames in `handle_missing_values()` and ensuring `normalize_city_names()` handles case variations correctly. This practice improved code reliability.

3. **Logging Integration:** Adding Python's `logging` module to every function from the start provided visibility into what each step does. The logs show exactly how many rows were removed, what values were imputed, and where the pipeline is at any moment. This makes debugging straightforward.

4. **Clear Project Structure:** The folder structure (src/, tests/, docs/, outputs/) was established early, making it easy to navigate and maintain the codebase. Having `__init__.py` files in packages enabled clean imports.

---

## What Could Be Improved

1. **Lack of Integration Tests:** While individual unit tests cover each function, there are no end-to-end integration tests that verify the entire pipeline from data loading through cleaning to EDA. In Sprint 2, the pipeline will extend to feature engineering and model training, making integration tests even more critical. 
   
   **Action:** Add integration tests in Sprint 2 that test the full pipeline from raw data to model predictions.

2. **Hardcoded Column Names:** Several functions reference specific column names like `"Base Fare"`, `"Tax & Surcharge"`, and `"Total Fare"` as string literals. If the dataset schema changes, multiple files would need to be updated.
   
   **Action:** Consider creating a configuration dictionary or constants file for column mappings in Sprint 2 to centralize these references.

---

## Action Items for Sprint 2

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | Add end-to-end integration tests for the full pipeline | Developer | High |
| 2 | Centralize column name references for maintainability | Developer | Medium |
| 3 | Add monitoring and logging enhancements to the ML pipeline | Developer | High |
| 4 | Document feature engineering and model results | Developer | Medium |

---

## Team Sentiment
Confident heading into Sprint 2. The data foundation is solid, the CI/CD pipeline is in place, and the code is well-tested. The main challenge in Sprint 2 will be managing the larger scope (5 user stories, 19 story points) while maintaining code quality.

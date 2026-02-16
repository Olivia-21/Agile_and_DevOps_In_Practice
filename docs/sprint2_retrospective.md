# Sprint 2 Retrospective (Final)

## Sprint Overview
- **Sprint Goal:** Build, optimize, and interpret machine learning models for flight fare prediction, and add monitoring/logging to the pipeline.
- **Planned Story Points:** 19
- **Delivered Story Points:** 19
- **Stories Completed:** US-04, US-05, US-06, US-07, US-08

---

## What Went Well

1. **Pipeline Modularity Paid Off:** The modular design established in Sprint 1 proved invaluable. The feature engineering module seamlessly consumed the output from the data cleaner, and the model training module could work with any feature-engineered dataset. Adding new models (Ridge, Lasso, RandomForest) was as simple as adding entries to a dictionary -- no structural changes required.

2. **Feedback-Driven Improvements:** The two improvements identified in the Sprint 1 Retrospective were both addressed:
   - Integration tests were added to validate the full pipeline end-to-end
   - Monitoring was implemented through the `PipelineMonitor` class, providing execution timing and step-level metrics
   
   This demonstrates the Agile principle of continuous improvement through retrospectives.

3. **Cross-Validation for Robust Evaluation:** Using 5-fold cross-validation alongside test-set metrics gave much more confidence in model performance estimates. This helped identify models that performed consistently across different data subsets rather than just on a single test split.

4. **Comprehensive Model Comparison:** Training five different regression models and comparing them in a standardized table made it easy to identify the best approach. The structured `create_comparison_table()` function produced clear, actionable output that could be shared with stakeholders.

---

## What Could Be Improved

1. **Dataset Size Limitations:** The current pipeline was tested with the Bangladesh flight price dataset, which may be limited in size. For production use, the pipeline should be tested with larger datasets to ensure scalability. Techniques like batch processing or Dask integration could be considered.

2. **Model Persistence:** Currently, trained models exist only in memory during script execution. In a real-world scenario, models should be serialized (using `joblib` or `pickle`) and versioned for deployment. This was not in the current scope but would be an important next step.

3. **Automated Reporting:** While the code generates plots and metrics, the reporting process (creating sprint review documents) was manual. In future projects, automated report generation -- possibly using Jupyter notebooks with papermill or Python templating -- would save time and reduce errors.

---

## Key Lessons Learned

### 1. Agile Practices
- **Sprint Planning is essential:** Breaking the project into sprints with clear goals prevented scope creep. Each sprint had a focused theme -- Sprint 1 on data foundation, Sprint 2 on modeling.
- **User Stories drive development:** Writing acceptance criteria before coding ensured every function had a clear purpose and testable outcome.
- **Retrospectives enable improvement:** The Sprint 1 retrospective directly led to integration testing and monitoring improvements in Sprint 2.

### 2. DevOps Practices
- **CI/CD catches issues early:** The GitHub Actions pipeline running flake8 and pytest on every push prevented code quality degradation. Linting caught formatting issues that would have accumulated over time.
- **Tests give confidence to refactor:** Having unit tests made it safe to refactor the feature engineering pipeline without worrying about breaking existing functionality.
- **Logging is not optional:** Integrated logging from day one made debugging pipeline issues straightforward. The `PipelineMonitor` class added in Sprint 2 elevated this to structured performance tracking.

### 3. Technical Insights
- **Feature engineering matters more than model complexity:** Date-derived features (Month, Season, Weekday) and proper encoding of categorical variables had a significant impact on model performance -- often more than switching between algorithms.
- **Cross-validation reveals model stability:** A model with a slightly lower R-squared but much lower standard deviation in cross-validation is often preferable to one with a higher R-squared but high variance.
- **Modular code is faster to develop:** While writing individual functions with docstrings and logging felt slower initially, it dramatically sped up Sprint 2 development since modules composed naturally.

---

## Project Metrics Summary

| Metric | Sprint 1 | Sprint 2 | Total |
|--------|----------|----------|-------|
| Story Points Planned | 12 | 19 | 31 |
| Story Points Delivered | 12 | 19 | 31 |
| Velocity | 100% | 100% | 100% |
| User Stories Completed | 3 | 5 | 8 |
| Test Files Created | 3 | 1 | 4 |
| Source Modules | 3 | 3 | 6 |

---

## Final Reflection

This project successfully demonstrated the application of Agile and DevOps practices to a real-world data science problem. The sprint-based approach provided structure without being rigid -- each sprint built naturally on the previous one. The DevOps practices (Git, CI/CD, testing, monitoring) transformed what could have been a fragile notebook-based analysis into a robust, maintainable, and reproducible ML pipeline.

The most impactful lesson was the synergy between Agile planning and DevOps execution: Agile ensured the right things were built, while DevOps ensured they were built correctly and reliably.

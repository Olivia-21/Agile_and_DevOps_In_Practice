"""
monitoring.py - Pipeline Monitoring & Logging

This module provides monitoring capabilities for the ML pipeline,
including execution time tracking, step-level metrics, and
structured error logging.

Sprint 2 Deliverable: Monitoring & Logging Enhancement
"""

import time
import logging
from datetime import datetime
from functools import wraps

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline_monitor")


class PipelineMonitor:
    """
    Monitor and track the performance of pipeline steps.

    Tracks execution time, status, and metrics for each step of the
    data science pipeline. Provides a summary report of pipeline health.

    Attributes
    ----------
    pipeline_name : str
        Name of the pipeline being monitored.
    steps : list
        List of recorded pipeline step metrics.
    start_time : float
        Timestamp when the pipeline started.
    errors : list
        List of recorded errors during pipeline execution.

    Example
    -------
    >>> monitor = PipelineMonitor("Flight Fare Pipeline")
    >>> monitor.start_pipeline()
    >>> monitor.log_step("Data Loading", status="success", rows=5000, cols=10)
    >>> monitor.log_step("Data Cleaning", status="success", rows_removed=120)
    >>> monitor.end_pipeline()
    >>> monitor.print_summary()
    """

    def __init__(self, pipeline_name: str = "ML Pipeline"):
        """
        Initialize the PipelineMonitor.

        Parameters
        ----------
        pipeline_name : str
            Name identifier for the pipeline.
        """
        self.pipeline_name = pipeline_name
        self.steps = []
        self.start_time = None
        self.end_time = None
        self.errors = []
        logger.info(f"PipelineMonitor initialized for: {pipeline_name}")

    def start_pipeline(self) -> None:
        """Record the start of the pipeline execution."""
        self.start_time = time.time()
        self.steps = []
        self.errors = []
        logger.info(f"Pipeline '{self.pipeline_name}' started at {datetime.now().isoformat()}")

    def end_pipeline(self) -> None:
        """Record the end of the pipeline execution."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(
            f"Pipeline '{self.pipeline_name}' completed in {duration:.2f} seconds "
            f"| Steps: {len(self.steps)} | Errors: {len(self.errors)}"
        )

    def log_step(self, step_name: str, status: str = "success", **metrics) -> None:
        """
        Log a pipeline step with its status and metrics.

        Parameters
        ----------
        step_name : str
            Name of the pipeline step.
        status : str
            Status of the step ('success', 'warning', 'error').
        **metrics
            Additional key-value metrics to log for this step.
        """
        step_record = {
            "step_name": step_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time if self.start_time else 0,
            "metrics": metrics,
        }
        self.steps.append(step_record)

        log_msg = f"Step '{step_name}': {status}"
        if metrics:
            metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            log_msg += f" | {metric_str}"

        if status == "error":
            logger.error(log_msg)
        elif status == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def log_error(self, step_name: str, error: Exception) -> None:
        """
        Log an error that occurred during a pipeline step.

        Parameters
        ----------
        step_name : str
            Name of the step where the error occurred.
        error : Exception
            The exception that was raised.
        """
        error_record = {
            "step_name": step_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        self.errors.append(error_record)
        logger.error(f"Error in '{step_name}': {type(error).__name__} - {error}")

    def get_summary(self) -> dict:
        """
        Generate a summary of the pipeline execution.

        Returns
        -------
        dict
            Summary containing pipeline name, duration, step count,
            error count, and step details.
        """
        total_duration = (
            (self.end_time - self.start_time) if self.end_time and self.start_time else None
        )

        return {
            "pipeline_name": self.pipeline_name,
            "total_duration_seconds": round(total_duration, 2) if total_duration else None,
            "total_steps": len(self.steps),
            "successful_steps": sum(1 for s in self.steps if s["status"] == "success"),
            "failed_steps": sum(1 for s in self.steps if s["status"] == "error"),
            "warnings": sum(1 for s in self.steps if s["status"] == "warning"),
            "errors": self.errors,
            "steps": self.steps,
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the pipeline execution."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print(f"  PIPELINE EXECUTION SUMMARY: {summary['pipeline_name']}")
        print("=" * 70)

        if summary["total_duration_seconds"]:
            print(f"  Total Duration: {summary['total_duration_seconds']:.2f} seconds")

        print(f"  Total Steps:    {summary['total_steps']}")
        print(f"  Successful:     {summary['successful_steps']} [OK]")
        print(f"  Warnings:       {summary['warnings']} [WARN]")
        print(f"  Failed:         {summary['failed_steps']} [FAIL]")

        print("\n  --- Step Details ---")
        print(f"  {'#':<4} {'Step':<30} {'Status':<10} {'Time (s)':<10}")
        print("  " + "-" * 58)

        for i, step in enumerate(summary["steps"], 1):
            status_icon = {"success": "[OK]", "warning": "[WARN]", "error": "[FAIL]"}.get(
                step["status"], "?"
            )
            print(
                f"  {i:<4} {step['step_name']:<30} {status_icon:<10} "
                f"{step['elapsed_time']:.2f}"
            )

            if step["metrics"]:
                for key, value in step["metrics"].items():
                    print(f"       |- {key}: {value}")

        if summary["errors"]:
            print("\n  --- Errors ---")
            for err in summary["errors"]:
                print(f"  [FAIL] [{err['step_name']}] {err['error_type']}: {err['error_message']}")

        print("=" * 70 + "\n")

    def health_check(self) -> str:
        """
        Return a health status based on pipeline execution.

        Returns
        -------
        str
            Health status: 'HEALTHY', 'DEGRADED', or 'UNHEALTHY'.
        """
        if not self.steps:
            return "NOT_STARTED"

        failed = sum(1 for s in self.steps if s["status"] == "error")
        warnings = sum(1 for s in self.steps if s["status"] == "warning")

        if failed > 0:
            status = "UNHEALTHY"
        elif warnings > 0:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        logger.info(f"Health check: {status} (failed={failed}, warnings={warnings})")
        return status


def monitor_step(step_name: str):
    """
    Decorator to automatically monitor a pipeline step.

    Parameters
    ----------
    step_name : str
        Name for this pipeline step.

    Returns
    -------
    function
        Decorated function with monitoring.

    Example
    -------
    >>> @monitor_step("Data Loading")
    ... def load_data(filepath):
    ...     return pd.read_csv(filepath)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"[MONITOR] Starting step: {step_name}")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(
                    f"[MONITOR] Step '{step_name}' completed successfully "
                    f"in {elapsed:.2f}s"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(
                    f"[MONITOR] Step '{step_name}' FAILED after {elapsed:.2f}s: "
                    f"{type(e).__name__} - {e}"
                )
                raise

        return wrapper

    return decorator

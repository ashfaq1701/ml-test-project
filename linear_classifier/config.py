"""Configuration and logging utilities for the linear classifier pipeline."""
from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

# Default dataset and output locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "result_linear_classification.csv"


DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: Optional[int] = None) -> None:
    """Configure application-wide logging if it is not already set.

    Parameters
    ----------
    level: Optional[int]
        Logging verbosity; defaults to ``logging.INFO``.
    """
    resolved_level = level or DEFAULT_LOG_LEVEL
    if logging.getLogger().handlers:
        # Avoid configuring logging multiple times when imported from notebooks.
        logging.getLogger().setLevel(resolved_level)
        return

    logging.basicConfig(
        level=resolved_level,
        format=LOG_FORMAT,
    )


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "TRAIN_PATH",
    "TEST_PATH",
    "RESULTS_DIR",
    "RESULTS_FILE",
    "DEFAULT_LOG_LEVEL",
    "LOG_FORMAT",
    "configure_logging",
]

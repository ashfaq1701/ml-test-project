"""Configuration for the DistilBERT classification pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "result_distilbert_classification.csv"

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1.0
CV_FOLDS = 3
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: Optional[int] = None) -> None:
    """Configure logging once for the application."""
    resolved_level = level or DEFAULT_LOG_LEVEL
    if logging.getLogger().handlers:
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
    "MODEL_NAME",
    "MAX_SEQ_LENGTH",
    "BATCH_SIZE",
    "GRADIENT_ACCUMULATION_STEPS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "NUM_EPOCHS",
    "CV_FOLDS",
    "VALIDATION_SPLIT",
    "RANDOM_STATE",
    "configure_logging",
]

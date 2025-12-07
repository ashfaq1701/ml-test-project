"""Data loading utilities for the linear classifier workflow."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetPaths:
    """Container for train and test dataset paths."""

    train: Path
    test: Path


@dataclass
class Dataset:
    """In-memory representation of the training and test sets."""

    train: pd.DataFrame
    test: pd.DataFrame


def load_datasets(paths: DatasetPaths) -> Dataset:
    """Load train and test CSVs and log their shapes."""
    logger.info("Loading datasets from %s and %s", paths.train, paths.test)
    train_df = pd.read_csv(paths.train)
    test_df = pd.read_csv(paths.test)

    for required in ("text", "author"):
        if required not in train_df.columns:
            raise ValueError(f"Expected '{required}' column in training data")
    if "text" not in test_df.columns:
        raise ValueError("Expected 'text' column in test data")

    logger.info(
        "Finished loading datasets: train=%d rows (%d cols), test=%d rows (%d cols)",
        len(train_df),
        len(train_df.columns),
        len(test_df),
        len(test_df.columns),
    )
    label_counts = train_df["author"].value_counts().to_dict()
    logger.info("Class distribution: %s", label_counts)
    return Dataset(train=train_df.reset_index(drop=True), test=test_df.reset_index(drop=True))


__all__ = ["DatasetPaths", "Dataset", "load_datasets"]

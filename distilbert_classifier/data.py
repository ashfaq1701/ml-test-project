"""Dataset helpers for DistilBERT fine-tuning."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset

from .config import RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class DatasetPaths:
    """Paths to training and test CSV files."""

    train: Path
    test: Path


@dataclass
class DatasetBundle:
    """In-memory representation of loaded data."""

    train: pd.DataFrame
    test: pd.DataFrame
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def load_datasets(paths: DatasetPaths) -> DatasetBundle:
    """Load and shuffle datasets, building label mappings."""
    logger.info("Loading datasets from %s and %s", paths.train, paths.test)
    train_df = pd.read_csv(paths.train)
    test_df = pd.read_csv(paths.test)

    if "text" not in train_df.columns or "author" not in train_df.columns:
        raise ValueError("Training data must include 'text' and 'author' columns")
    if "text" not in test_df.columns:
        raise ValueError("Test data must include a 'text' column")

    logger.info("Shuffling training data with random state %d", RANDOM_STATE)
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    labels: List[str] = sorted(train_df["author"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    train_df = train_df.assign(labels=train_df["author"].map(label2id))

    logger.info(
        "Finished loading datasets: train=%d rows | test=%d rows | classes=%d",
        len(train_df),
        len(test_df),
        len(label2id),
    )
    return DatasetBundle(train=train_df, test=test_df, label2id=label2id, id2label=id2label)


def to_hf_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Dataset, Dataset]:
    """Convert pandas DataFrames to Hugging Face Datasets."""
    train_hf = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    test_hf = Dataset.from_pandas(test_df[["text"]], preserve_index=False)
    return train_hf, test_hf


__all__ = ["DatasetPaths", "DatasetBundle", "load_datasets", "to_hf_dataset"]

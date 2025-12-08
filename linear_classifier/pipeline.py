"""End-to-end workflow for the linear classifier."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import RESULTS_FILE, TEST_PATH, TRAIN_PATH, configure_logging, SAVED_MODELS_PATH
from .data import DatasetPaths, load_datasets
from .features import build_vectorizer
from .model import CV_FOLDS, build_model, evaluate_model, fit_model

logger = logging.getLogger(__name__)


RESULTS_FILENAME = RESULTS_FILE.name


def run_linear_pipeline() -> Path:
    """Execute the full pipeline: load data, validate, train, and export predictions."""
    configure_logging()
    logger.info("Initializing linear classification workflow")

    dataset_paths = DatasetPaths(train=TRAIN_PATH, test=TEST_PATH)
    dataset = load_datasets(dataset_paths)
    logger.info("Training samples: %d | Test samples: %d", len(dataset.train), len(dataset.test))

    vectorizer = build_vectorizer()
    pipeline = build_model(vectorizer)

    X_train = dataset.train["text"]
    y_train = dataset.train["author"]

    logger.info("Running %d-fold cross-validation", CV_FOLDS)
    evaluate_model(pipeline, X_train, y_train)

    trained_pipeline = fit_model(pipeline, X_train, y_train)

    joblib.dump(trained_pipeline, SAVED_MODELS_PATH)
    logger.info("Saved trained model to %s", SAVED_MODELS_PATH)

    predictions = trained_pipeline.predict(dataset.test["text"])

    results_df = pd.DataFrame({"author": predictions})
    results_file = _save_results(results_df, RESULTS_FILE)
    logger.info("Saved predictions to %s", results_file)
    return results_file


def _save_results(results_df: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(destination, index=False)
    return destination


__all__ = ["run_linear_pipeline", "RESULTS_FILENAME"]

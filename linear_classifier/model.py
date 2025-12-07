"""Model construction and evaluation helpers."""
from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


DEFAULT_C = 1.0
CV_FOLDS = 5
RANDOM_STATE = 42


def build_model(vectorizer: object) -> Pipeline:
    """Compose the TF-IDF vectorizer with a linear SVM classifier."""
    classifier = LinearSVC(
        C=DEFAULT_C,
        dual=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])


def evaluate_model(pipeline: Pipeline, X: Iterable[str], y: Iterable[str]) -> Tuple[float, float]:
    """Run cross-validation and return mean and standard deviation of accuracy."""
    y_list = list(y)
    logger.info("Starting %d-fold cross-validation on %d samples", CV_FOLDS, len(y_list))
    scores = cross_val_score(
        pipeline,
        X,
        y_list,
        cv=CV_FOLDS,
        n_jobs=-1,
        scoring="accuracy",
    )
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    logger.info("Cross-validation accuracy: %.4f ± %.4f", mean_score, std_score)
    return mean_score, std_score


def fit_model(pipeline: Pipeline, X: Iterable[str], y: Iterable[str]) -> Pipeline:
    """Fit the pipeline on the full training data."""
    logger.info("Training final model on full dataset")
    return pipeline.fit(X, y)


__all__ = [
    "build_model",
    "evaluate_model",
    "fit_model",
    "CV_FOLDS",
    "RANDOM_STATE",
    "DEFAULT_C",
]

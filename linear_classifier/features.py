"""Feature engineering for text classification."""
from __future__ import annotations

import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


logger = logging.getLogger(__name__)


CHAR_NGRAM_RANGE = (3, 5)
MAX_CHAR_FEATURES = 5000

WORD_NGRAM_RANGE = (1, 2)
MAX_WORD_FEATURES = 20000


def _build_char_vectorizer() -> TfidfVectorizer:
    logger.info(
        "Building character TF-IDF | ngram_range=%s | max_features=%d",
        CHAR_NGRAM_RANGE,
        MAX_CHAR_FEATURES,
    )
    return TfidfVectorizer(
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=MAX_CHAR_FEATURES,
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        analyzer="char",
    )


def _build_word_vectorizer() -> TfidfVectorizer:
    logger.info(
        "Building word TF-IDF | ngram_range=%s | max_features=%d",
        WORD_NGRAM_RANGE,
        MAX_WORD_FEATURES,
    )
    return TfidfVectorizer(
        ngram_range=WORD_NGRAM_RANGE,
        max_features=MAX_WORD_FEATURES,
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        analyzer="word",
    )


def build_vectorizer() -> Any:
    """Create a combined word + character TF-IDF feature space."""
    logger.info("Combining word and character TF-IDF vectorizers via FeatureUnion")
    return FeatureUnion([
        ("word_tfidf", _build_word_vectorizer()),
        ("char_tfidf", _build_char_vectorizer()),
    ])


__all__ = [
    "build_vectorizer",
    "CHAR_NGRAM_RANGE",
    "MAX_CHAR_FEATURES",
    "WORD_NGRAM_RANGE",
    "MAX_WORD_FEATURES",
]

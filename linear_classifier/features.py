"""Feature engineering for text classification."""
from __future__ import annotations

import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


CHAR_NGRAM_RANGE = (3, 5)
MAX_CHAR_FEATURES = 5000


def build_vectorizer() -> Any:
    """Create a character-level TF-IDF vectorizer for stylistic signals."""
    logger.info(
        "Building character TF-IDF vectorizer | ngram_range=%s | max_features=%d",
        CHAR_NGRAM_RANGE,
        MAX_CHAR_FEATURES,
    )
    vectorizer = TfidfVectorizer(
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=MAX_CHAR_FEATURES,
        min_df=2,
        sublinear_tf=True,
        lowercase=True,
        analyzer="char",
    )
    return vectorizer


__all__ = [
    "build_vectorizer",
    "CHAR_NGRAM_RANGE",
    "MAX_CHAR_FEATURES",
]

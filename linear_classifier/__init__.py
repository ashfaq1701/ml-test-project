"""Linear classifier package with TF-IDF features and linear SVM training."""
from .pipeline import run_linear_pipeline, RESULTS_FILENAME

__all__ = ["run_linear_pipeline", "RESULTS_FILENAME"]

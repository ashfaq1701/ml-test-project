"""End-to-end workflow for DistilBERT fine-tuning."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import DatasetDict

from .config import (
    RESULTS_FILE,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_SPLIT,
    RANDOM_STATE,
    configure_logging, SAVED_MODELS_DIR,
)
from .data import DatasetPaths, load_datasets, to_hf_dataset
from .model import (
    build_tokenizer,
    predict_labels,
    run_cross_validation,
    tokenize_dataset,
    train_final_model,
)

logger = logging.getLogger(__name__)

RESULTS_FILENAME = RESULTS_FILE.name


def run_distilbert_pipeline() -> Path:
    """Run the DistilBERT workflow: load data, cross-validate, train, predict."""
    configure_logging()
    logger.info("Initializing DistilBERT classification workflow")

    dataset_paths = DatasetPaths(train=TRAIN_PATH, test=TEST_PATH)
    dataset_bundle = load_datasets(dataset_paths)

    train_hf, test_hf = to_hf_dataset(dataset_bundle.train, dataset_bundle.test)
    tokenizer = build_tokenizer()
    tokenized_train = tokenize_dataset(train_hf, tokenizer)
    tokenized_test = tokenize_dataset(test_hf, tokenizer)

    run_cross_validation(
        tokenized_train,
        label_list=sorted(dataset_bundle.label2id.keys()),
        output_dir=RESULTS_FILE.parent / "distilbert_cv",
    )

    dataset_splits = _train_eval_split(tokenized_train)
    trainer = train_final_model(
        tokenized_train=dataset_splits["train"],
        tokenized_eval=dataset_splits["test"],
        label2id=dataset_bundle.label2id,
        output_dir=RESULTS_FILE.parent / "distilbert_final",
    )

    trainer.save_model(SAVED_MODELS_DIR)  # saves model weights + config
    tokenizer.save_pretrained(SAVED_MODELS_DIR)

    logger.info("Saved trained model to %s", SAVED_MODELS_DIR)

    predictions = predict_labels(trainer, tokenized_test, dataset_bundle.id2label)

    results_df = pd.DataFrame({"author": predictions})
    results_file = _save_results(results_df, RESULTS_FILE)
    logger.info("Saved predictions to %s", results_file)
    return results_file


def _train_eval_split(tokenized_train) -> DatasetDict:
    """Create a deterministic train/eval split for final training."""
    return tokenized_train.train_test_split(test_size=VALIDATION_SPLIT, seed=RANDOM_STATE)


def _save_results(results_df: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(destination, index=False)
    return destination


__all__ = ["run_distilbert_pipeline", "RESULTS_FILENAME"]

"""Model builders and training utilities for DistilBERT."""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import (
    BATCH_SIZE,
    CV_FOLDS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    RANDOM_STATE,
    WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


def build_tokenizer() -> AutoTokenizer:
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def build_model(label2id: Dict[str, int]) -> AutoModelForSequenceClassification:
    logger.info("Loading model: %s", MODEL_NAME)
    num_labels = len(label2id)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={idx: label for label, idx in label2id.items()},
        label2id=label2id,
    )


def build_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    logger.info("Tokenizing dataset with max_length=%d", MAX_SEQ_LENGTH)

    def _tokenize(batch: Dict[str, Iterable[str]]):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    remove_columns = [col for col in dataset.column_names if col != "labels"]
    return dataset.map(_tokenize, batched=True, remove_columns=remove_columns)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    return {"accuracy": accuracy}


def _training_args(output_dir: Path, evaluation: bool = True) -> TrainingArguments:
    params = inspect.signature(TrainingArguments).parameters

    required_args = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_train_epochs": NUM_EPOCHS,
    }

    optional_args = {
        "evaluation_strategy": "epoch" if evaluation else "no",
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "load_best_model_at_end": False,
        "report_to": [],
    }

    for key, value in optional_args.items():
        if key in params:
            required_args[key] = value

    return TrainingArguments(**required_args)


def run_cross_validation(
    tokenized_train: Dataset, label_list: Sequence[str], output_dir: Path
) -> List[float]:
    """Perform stratified cross-validation and return per-fold accuracies."""
    logger.info("Starting %d-fold cross-validation", CV_FOLDS)
    set_seed(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_indices = np.arange(len(tokenized_train))
    labels = np.array(tokenized_train["labels"])
    fold_scores: List[float] = []

    for fold_index, (train_idx, eval_idx) in enumerate(skf.split(all_indices, labels), start=1):
        logger.info("Fold %d/%d | train=%d | eval=%d", fold_index, CV_FOLDS, len(train_idx), len(eval_idx))
        fold_train = tokenized_train.select(train_idx.tolist())
        fold_eval = tokenized_train.select(eval_idx.tolist())

        tokenizer = build_tokenizer()
        model = build_model({label: i for i, label in enumerate(label_list)})
        data_collator = build_data_collator(tokenizer)

        trainer = Trainer(
            model=model,
            args=_training_args(output_dir / f"fold_{fold_index}", evaluation=True),
            train_dataset=fold_train,
            eval_dataset=fold_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        accuracy = float(metrics.get("eval_accuracy", 0.0))
        logger.info("Fold %d accuracy: %.4f", fold_index, accuracy)
        fold_scores.append(accuracy)

    mean_acc = float(np.mean(fold_scores))
    std_acc = float(np.std(fold_scores))
    logger.info("Cross-validation accuracy: %.4f ± %.4f", mean_acc, std_acc)
    return fold_scores


def train_final_model(
    tokenized_train: Dataset,
    tokenized_eval: Dataset,
    label2id: Dict[str, int],
    output_dir: Path,
) -> Trainer:
    """Train the final model on the full training data (with a held-out eval set)."""
    set_seed(RANDOM_STATE)
    tokenizer = build_tokenizer()
    model = build_model(label2id)
    data_collator = build_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=_training_args(output_dir, evaluation=True),
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    logger.info("Final model eval metrics: %s", eval_metrics)
    return trainer


def predict_labels(trainer: Trainer, tokenized_test: Dataset, id2label: Dict[int, str]) -> List[str]:
    logger.info("Generating predictions for test set")
    predictions = trainer.predict(tokenized_test)
    label_ids = np.argmax(predictions.predictions, axis=-1)
    return [id2label[int(idx)] for idx in label_ids]


__all__ = [
    "build_tokenizer",
    "build_model",
    "build_data_collator",
    "tokenize_dataset",
    "run_cross_validation",
    "train_final_model",
    "predict_labels",
]

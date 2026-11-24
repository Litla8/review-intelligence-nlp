"""Step 6 helpers for the DistilBERT experiment.

Everything here works on plain NumPy arrays and DataFrames, so it is tested without a GPU. The code that needs
PyTorch and Hugging Face lives in ``reviews/train_transformer.py`` and imports them only when it runs.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from reviews.config import TransformerConfig

LOGGER = logging.getLogger(__name__)


class EncodedDataset:
    """Tokenised reviews as a map-style dataset (``len`` and ``[i]``), which is all the Trainer's DataLoader needs.

    Padding is left to the data collator, so every batch is padded only to its own longest review.
    """

    def __init__(self, encodings: Mapping[str, Sequence[Sequence[int]]], labels: Sequence[int]) -> None:
        sizes = {len(v) for v in encodings.values()}
        if sizes != {len(labels)}:
            raise ValueError("every encoding field must have exactly one entry per label")
        self._encodings = {key: list(value) for key, value in encodings.items()}
        self._labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {key: value[index] for key, value in self._encodings.items()}
        item["labels"] = self._labels[index]
        return item


def label_ids(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    """Map class names to 0/1 ids in the order of ``classes``; unknown labels raise."""
    lookup = {name: i for i, name in enumerate(classes)}
    unknown = set(labels) - set(lookup)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    return np.array([lookup[x] for x in labels], dtype=np.int64)


def positive_probability(logits: np.ndarray) -> np.ndarray:
    """Softmax over two logits, returning the probability of class 1 (positive), computed stably."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("logits must have shape (n_reviews, 2)")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1] / exp.sum(axis=1)


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Metrics for the Trainer after each epoch: accuracy and macro-F1 (the metric used to keep the best checkpoint)."""
    logits, labels = eval_pred
    predicted = np.asarray(logits).argmax(axis=-1)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0))}


def best_threshold(y_true: np.ndarray, p_positive: np.ndarray, cfg: TransformerConfig = TransformerConfig()) -> Tuple[float, float]:
    """The probability cut-off for "positive" with the best macro-F1, searched on the validation set only.

    The default 0.5 is not optimal when 84% of reviews are positive. Ties go to the value closest to 0.5.
    """
    y_true = np.asarray(y_true)
    grid = np.round(np.arange(cfg.threshold_min, cfg.threshold_max + 1e-9, cfg.threshold_step), 4)
    scores = np.array([f1_score(y_true, (p_positive >= t).astype(int), average="macro", zero_division=0) for t in grid])
    best = scores.max()
    candidates = grid[np.isclose(scores, best)]
    chosen = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return chosen, float(best)


def apply_threshold(p_positive: np.ndarray, threshold: float) -> np.ndarray:
    """0/1 predictions: 1 (positive) when the probability reaches ``threshold``."""
    return (np.asarray(p_positive) >= threshold).astype(np.int64)


def slice_scores(test: pd.DataFrame, predictions: Mapping[str, np.ndarray], y_true: np.ndarray,
                 fraction: float) -> Dict[str, Any]:
    """Macro-F1 of each model on the oldest and newest ``fraction`` of the test set (by review date).

    The models were not trained on the test set, so this shows whether newer wording hurts without retraining.
    """
    if "review_date" not in test.columns:
        raise ValueError("test frame needs a review_date column")
    n = len(test)
    k = max(1, int(round(n * fraction)))
    order = np.argsort(test["review_date"].to_numpy(), kind="stable")
    slices = {"oldest": order[:k], "newest": order[n - k:]}
    out: Dict[str, Any] = {"reviews_per_slice": int(k)}
    for name, idx in slices.items():
        dates = test["review_date"].iloc[idx]
        out[name] = {"from": str(dates.min().date()), "to": str(dates.max().date()),
                     "positive_share": round(float(np.mean(y_true[idx])), 4),
                     "macro_f1": {model: round(float(f1_score(y_true[idx], pred[idx], average="macro", zero_division=0)), 4)
                                  for model, pred in predictions.items()}}
    return out


def confident_errors(test: pd.DataFrame, p_positive: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray,
                     classes: Sequence[str], limit: int) -> pd.DataFrame:
    """The ``limit`` wrong predictions the model was most sure about, with the original review text."""
    wrong = y_pred != y_true
    confidence = np.where(y_pred == 1, p_positive, 1 - p_positive)
    cols = [c for c in ("Id", "Score", "sentiment", "Summary", "Text") if c in test.columns]
    out = test.loc[wrong, cols].copy()
    out["predicted"] = np.asarray(classes)[y_pred[wrong]]
    out["confidence"] = confidence[wrong].round(4)
    return out.sort_values("confidence", ascending=False, kind="stable").head(limit).reset_index(drop=True)


def combine_tables(classical: Optional[pd.DataFrame], transformer_row: Mapping[str, Any]) -> pd.DataFrame:
    """Add the transformer's row to the Step 5 comparison table, best test macro-F1 first."""
    rows = [] if classical is None else classical.to_dict(orient="records")
    rows = [r for r in rows if r.get("model") != transformer_row["model"]]  # rerunning replaces the old row
    table = pd.DataFrame(rows + [dict(transformer_row)])
    return table.sort_values("test_macro_f1", ascending=False, kind="stable").reset_index(drop=True)


def gpu_message(has_cuda: bool, name: Optional[str], torch_version: str) -> str:
    """A one-line description of the training device, or the fix when PyTorch cannot see the GPU."""
    if has_cuda:
        return f"training on {name} (torch {torch_version})"
    return (f"PyTorch {torch_version} cannot see a CUDA GPU. For an RTX 50-series card install the CUDA 12.8 build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall")

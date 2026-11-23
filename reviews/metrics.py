"""Classification and topic-model metrics, plus the success-target check."""
from __future__ import annotations

import time
from typing import Callable, Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from reviews.config import TargetConfig


def _check(y_true: Sequence, y_pred: Sequence) -> None:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    """Unweighted mean F1 over classes; the primary metric because minority classes matter as much."""
    _check(y_true, y_pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate_classifier(y_true: Sequence, y_pred: Sequence, classes: Sequence[str]) -> Dict[str, object]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix (rows = truth)."""
    _check(y_true, y_pred)
    labels = list(classes)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                      for c, p, r, f, s in zip(labels, precision, recall, f1, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": labels,
    }


def time_inference(predict: Callable[[Sequence[str]], Sequence], texts: Sequence[str], repeats: int = 3) -> Dict[str, float]:
    """Median wall-clock seconds per 1,000 reviews for a ``predict(texts)`` callable."""
    if not texts or repeats < 1:
        raise ValueError("Need at least one text and one repeat")
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(texts)
        durations.append(time.perf_counter() - start)
    median = float(np.median(durations))
    return {"seconds_total": median, "seconds_per_1000": median / len(texts) * 1000}


def check_targets(macro_f1_binary: float, coherence: float, targets: TargetConfig = TargetConfig()) -> Dict[str, bool]:
    """Compare results with the brief's success metrics (True means the target is met)."""
    return {
        "macro_f1_binary_above_target": macro_f1_binary > targets.macro_f1_binary,
        "topic_coherence_above_target": coherence > targets.topic_coherence,
    }


def worth_the_cost(baseline_macro_f1: float, model_macro_f1: float, targets: TargetConfig = TargetConfig()) -> bool:
    """True if the heavier model beats the baseline by at least the configured margin."""
    return model_macro_f1 - baseline_macro_f1 >= targets.min_gain_over_baseline

"""Turn star ratings into sentiment labels and check class balance."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from reviews.config import LabelConfig


def stars_to_sentiment(stars: pd.Series, cfg: LabelConfig = LabelConfig()) -> pd.Series:
    """Map 1-5 star ratings to ``negative`` / ``neutral`` / ``positive`` (vectorised).

    Raises:
        ValueError: If any rating is missing or outside 1-5.
    """
    values = pd.to_numeric(stars, errors="coerce")
    if values.isna().any() or not values.between(1, 5).all():
        raise ValueError("Star ratings must be numbers from 1 to 5 with no missing values")
    labels = np.select([values <= cfg.negative_max, values >= cfg.positive_min], list(cfg.classes[::2]), default=cfg.classes[1])
    return pd.Series(pd.Categorical(labels, categories=list(cfg.classes)), index=stars.index, name="sentiment")


def to_binary(sentiment: pd.Series, cfg: LabelConfig = LabelConfig()) -> pd.Series:
    """Drop neutral rows; the binary task (and the 0.88 macro-F1 target) is positive vs negative."""
    return sentiment[sentiment.isin(cfg.binary_classes)].astype(str)


def class_balance(sentiment: pd.Series, cfg: LabelConfig = LabelConfig()) -> Dict[str, Dict[str, float]]:
    """Count and share of each class, including empty classes, plus the majority-to-minority ratio."""
    counts = sentiment.value_counts().reindex(list(cfg.classes), fill_value=0)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("No labels to count")
    non_empty = counts[counts > 0]
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "share": {k: round(float(v) / total, 4) for k, v in counts.items()},
        "imbalance_ratio": {"majority_to_minority": round(float(non_empty.max() / non_empty.min()), 2)},
    }

"""Tests for reviews.predict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reviews import predict


def test_read_threshold(tmp_path: Path) -> None:
    assert predict.read_threshold(tmp_path) == 0.5          # missing file -> default
    (tmp_path / "inference_config.json").write_text(json.dumps({"threshold": 0.53}))
    assert predict.read_threshold(tmp_path) == 0.53


def test_label_probabilities_bands() -> None:
    out = predict.label_probabilities(np.array([0.05, 0.25, 0.5, 0.75, 0.99]), 0.5)
    assert out.tolist() == ["negative", "negative", "mixed", "positive", "positive"]


def test_load_scorer_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_transformer"):
        predict.load_scorer(tmp_path / "nope", 64, 8)

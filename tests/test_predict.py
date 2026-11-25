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


def test_resolve_model_dir(tmp_path: Path) -> None:
    from unittest import mock
    assert predict.resolve_model_dir(tmp_path) == tmp_path                      # exists: used as is
    with pytest.raises(FileNotFoundError, match="HF_MODEL_REPO"):
        predict.resolve_model_dir(tmp_path / "nope")
    fake_hub = mock.MagicMock()
    fake_hub.snapshot_download.return_value = str(tmp_path / "cache")
    with mock.patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
        assert predict.resolve_model_dir(tmp_path / "nope", "user/model") == tmp_path / "cache"
    fake_hub.snapshot_download.assert_called_once_with(repo_id="user/model")


def test_load_scorers_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        predict.load_scorers(tmp_path / "nope", [(64, 8), (128, 8)])

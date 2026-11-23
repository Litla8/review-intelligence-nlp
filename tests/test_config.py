"""Tests for reviews.config and logging."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from reviews.config import PathsConfig, ProjectConfig
from reviews.logging_utils import configure_logging


def test_paths_are_relative_to_root(tmp_path: Path) -> None:
    paths = PathsConfig(root=tmp_path)
    assert paths.raw_path == tmp_path / "data/raw/Reviews.csv" and paths.models_path == tmp_path / "models"


def test_defaults_match_the_brief() -> None:
    cfg = ProjectConfig()
    assert cfg.labels.classes == ("negative", "neutral", "positive")
    assert cfg.targets.macro_f1_binary == 0.88 and cfg.targets.topic_coherence == 0.5
    assert cfg.split.random_state == 42 and len(cfg.aspects.keywords) >= 6
    assert all(words for words in cfg.aspects.keywords.values())


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        ProjectConfig().split.random_state = 1  # type: ignore[misc]


def test_logging_rejects_unknown_level() -> None:
    configure_logging("debug")
    with pytest.raises(ValueError):
        configure_logging("loud")

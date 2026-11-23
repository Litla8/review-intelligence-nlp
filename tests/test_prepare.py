"""Tests for reviews.prepare."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from reviews.config import ProjectConfig
from reviews.data import clean_reviews
from reviews.prepare import build_quality_report, length_summary, main, run


def test_quality_report_counts(raw_reviews: pd.DataFrame, cfg: ProjectConfig) -> None:
    result = clean_reviews(raw_reviews, cfg.data)
    report = build_quality_report(len(raw_reviews), result, cfg)
    assert report["rows_raw"] == 19 and report["rows_clean"] == 10 and report["share_kept"] == round(10 / 19, 4)
    assert report["class_balance_three_class"]["counts"] == {"negative": 3, "neutral": 1, "positive": 6}
    assert report["binary_rows"] == 9 and report["binary_positive_share"] == round(6 / 9, 4)
    assert report["stars"] == {"1": 2, "2": 1, "3": 1, "4": 2, "5": 4}
    assert report["date_range"][0] < report["date_range"][1]
    assert sum(report["reviews_per_year"].values()) == 10
    json.dumps(report)  # must be serialisable


def test_length_summary_has_every_class(raw_reviews: pd.DataFrame, cfg: ProjectConfig) -> None:
    frame = clean_reviews(raw_reviews, cfg.data).frame
    summary = length_summary(frame)
    assert set(summary["by_sentiment"]) == {"negative", "neutral", "positive"}
    assert summary["all"]["max"] >= summary["all"]["median"]


def test_run_writes_both_outputs_and_is_repeatable(raw_reviews: pd.DataFrame, cfg: ProjectConfig) -> None:
    pytest.importorskip("pyarrow")
    cfg.paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_reviews.to_csv(cfg.paths.raw_path, index=False)
    first = run(cfg)
    clean = pd.read_parquet(cfg.paths.clean_path)
    assert len(clean) == 10 and clean["sentiment"].astype(str).isin({"negative", "neutral", "positive"}).all()
    assert json.loads(cfg.paths.quality_report_path.read_text(encoding="utf-8")) == first
    assert run(cfg) == first


def test_main_runs_from_a_root_folder(raw_reviews: pd.DataFrame, cfg: ProjectConfig) -> None:
    pytest.importorskip("pyarrow")
    cfg.paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_reviews.to_csv(cfg.paths.raw_path, index=False)
    assert main(["--root", str(cfg.paths.root), "--log-level", "WARNING"]) == 0
    assert cfg.paths.clean_path.is_file()


def test_main_without_the_raw_file_fails_clearly(cfg: ProjectConfig) -> None:
    with pytest.raises(FileNotFoundError, match="Reviews.csv"):
        main(["--root", str(cfg.paths.root), "--log-level", "WARNING"])

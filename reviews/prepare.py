"""Step 2 pipeline: load the raw reviews, clean and label them, write the cleaned table and a quality report.

Run from the project root::

    python -m reviews.prepare

Re-running gives the same output files (the pipeline is deterministic and overwrites).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from reviews.config import ProjectConfig
from reviews.data import CleaningResult, clean_reviews, load_raw
from reviews.labels import class_balance, to_binary
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def length_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    """Words per review overall and by sentiment: median, 95th percentile and maximum."""
    def describe(words: pd.Series) -> Dict[str, float]:
        return {"median": float(words.median()), "p95": float(words.quantile(0.95)), "max": int(words.max())}

    return {"all": describe(frame["n_words"]),
            "by_sentiment": {str(k): describe(g["n_words"]) for k, g in frame.groupby("sentiment", observed=True)}}


def build_quality_report(raw_rows: int, result: CleaningResult, cfg: ProjectConfig) -> Dict[str, Any]:
    """Everything worth checking about the cleaned data, as a JSON-serialisable dict."""
    frame = result.frame
    binary = to_binary(frame["sentiment"], cfg.labels)
    per_year = frame.groupby(frame["review_date"].dt.year).size()
    share_pos_by_year = frame.assign(pos=frame["sentiment"].eq("positive")).groupby(frame["review_date"].dt.year)["pos"].mean()
    return {
        "rows_raw": int(raw_rows),
        "rows_clean": int(len(frame)),
        "share_kept": round(len(frame) / raw_rows, 4),
        "cleaning_steps": result.stages,
        "invalid_helpfulness_votes_set_to_nan": result.invalid_helpfulness_votes,
        "date_range": [frame["review_date"].min().date().isoformat(), frame["review_date"].max().date().isoformat()],
        "unique_products": int(frame[cfg.data.product_column].nunique()),
        "unique_users": int(frame[cfg.data.user_column].nunique()),
        "class_balance_three_class": class_balance(frame["sentiment"], cfg.labels),
        "binary_rows": int(len(binary)),
        "binary_positive_share": round(float((binary == "positive").mean()), 4),
        "stars": {str(int(k)): int(v) for k, v in frame[cfg.data.rating_column].value_counts().sort_index().items()},
        "words_per_review": length_summary(frame),
        "reviews_per_year": {str(int(k)): int(v) for k, v in per_year.items()},
        "positive_share_per_year": {str(int(k)): round(float(v), 4) for k, v in share_pos_by_year.items()},
    }


def run(cfg: ProjectConfig) -> Dict[str, Any]:
    """Load, clean, label and write ``reviews_clean.parquet`` and ``data_quality.json``; return the report."""
    raw = load_raw(cfg.paths.raw_path, cfg.data)
    LOGGER.info("loaded %d raw reviews", len(raw))
    result = clean_reviews(raw, cfg.data, cfg.labels)
    report = build_quality_report(len(raw), result, cfg)
    cfg.paths.processed_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(cfg.paths.clean_path, index=False)
    cfg.paths.quality_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s and %s", cfg.paths.clean_path, cfg.paths.quality_report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Clean and label the raw review file (Step 2).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    from dataclasses import replace
    cfg = ProjectConfig()
    cfg = replace(cfg, paths=replace(cfg.paths, root=args.root))
    report = run(cfg)
    LOGGER.info("kept %d of %d reviews (%.1f%%)", report["rows_clean"], report["rows_raw"], 100 * report["share_kept"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

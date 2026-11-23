"""Step 4 pipeline: read the preprocessed reviews, write charts to ``reports/figures`` and ``reports/eda_summary.json``.

Run from the project root::

    python -m reviews.explore
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from reviews import eda, plots
from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def analyse(frame: pd.DataFrame, cfg: ProjectConfig, figures_dir: Path, cloud: Optional[Any] = None) -> Dict[str, Any]:
    """Compute every Step 4 table, save every chart and return the summary dictionary (no file reading or JSON here)."""
    eda.require_columns(frame, ["sentiment", "text_clean", "n_words", "n_tokens", "review_date", "Score"])
    e = cfg.eda
    counts = eda.class_word_counts(frame)
    shown = {name: eda.top_words_from_counts(c, e.top_n, e.display_stopwords) for name, c in counts.items()}
    plots.save_figure(plots.plot_length_distribution(frame, e.length_clip_quantile), figures_dir / "length_distribution.png", e.dpi)
    plots.save_figure(plots.plot_top_words(shown), figures_dir / "top_words_by_sentiment.png", e.dpi)
    cloud_counts = {name: c.drop(labels=list(set(e.display_stopwords) & set(c.index))) for name, c in counts.items()}
    plots.save_figure(plots.plot_wordclouds(cloud_counts, e.cloud_max_words, cloud), figures_dir / "wordclouds.png", e.dpi)

    sample = eda.sample_per_class(frame, e.ngram_max_docs)
    for n, label in ((2, "Bigrams"), (3, "Trigrams")):
        phrases = {str(name): eda.top_ngrams(group["text_clean"], n, e.ngram_top, e.display_stopwords)
                   for name, group in sample.groupby("sentiment", observed=True)}
        plots.save_figure(plots.plot_ngrams(phrases, label), figures_dir / f"top_{label.lower()}.png", e.dpi)

    summary = eda.build_summary(frame, e, counts)
    if "distinctive_negative" in summary:
        plots.save_figure(plots.plot_distinctive(pd.Series(summary["distinctive_negative"]),
                                                 pd.Series(summary["distinctive_positive"])),
                          figures_dir / "distinctive_words.png", e.dpi)
    monthly = eda.monthly_trend(frame, e.min_month_reviews)
    plots.save_figure(plots.plot_rating_trend(monthly), figures_dir / "sentiment_trend.png", e.dpi)
    summary["months_plotted"] = int(len(monthly))
    summary["ngram_sample_per_class"] = e.ngram_max_docs
    return summary


def run(cfg: ProjectConfig, cloud: Optional[Any] = None) -> Dict[str, Any]:
    """Read ``reviews_text.parquet`` and write charts and the JSON summary.

    Raises:
        FileNotFoundError: If Step 3 has not been run on the full data.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    frame = pd.read_parquet(source)
    LOGGER.info("exploring %d reviews", len(frame))
    summary = analyse(frame, cfg, cfg.paths.figures_path, cloud)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.eda_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("wrote charts to %s and %s", cfg.paths.figures_path, cfg.paths.eda_summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Exploratory text analysis (Step 4).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    cfg = replace(ProjectConfig(), paths=replace(ProjectConfig().paths, root=args.root))
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

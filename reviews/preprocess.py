"""Step 3 pipeline: read the cleaned reviews, add ``text_clean``, write the result and a report.

Run from the project root::

    python -m reviews.preprocess --sample       # 5,000 reviews, a minute or less, to check everything works
    python -m reviews.preprocess                # all reviews
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd

from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging
from reviews.text import load_spacy, preprocess, text_report

LOGGER = logging.getLogger(__name__)


def run(cfg: ProjectConfig, sample: bool = False, nlp_loader: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Preprocess the cleaned reviews and write ``reviews_text*.parquet`` and ``text_report.json``.

    Raises:
        FileNotFoundError: If Step 2 has not been run.
    """
    source = cfg.paths.clean_path
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.prepare` first.")
    frame = pd.read_parquet(source)
    if sample:
        frame = frame.sample(n=min(cfg.text.sample_rows, len(frame)), random_state=42).sort_index()  # same rows every run
    LOGGER.info("preprocessing %d reviews", len(frame))
    nlp = (nlp_loader or load_spacy)(cfg.text)  # looked up at call time so tests can replace it
    result = preprocess(frame, nlp, cfg.text, cfg.data.text_column)
    report = {"sample": sample, **text_report(result, cfg.text)}
    cfg.paths.processed_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cfg.paths.text_path(sample), index=False)
    report_path = cfg.paths.text_report_path if not sample else cfg.paths.reports_path / "text_report_sample.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s and %s", cfg.paths.text_path(sample), report_path)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Preprocess review text (Step 3).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="only a small random sample, written to separate files")
    parser.add_argument("--n-process", type=int, default=None, help="spaCy worker processes (1 = single process)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    cfg = ProjectConfig()
    text = cfg.text if args.n_process is None else replace(cfg.text, n_process=args.n_process)
    cfg = replace(cfg, paths=replace(cfg.paths, root=args.root), text=text)
    report = run(cfg, sample=args.sample)
    LOGGER.info("done: %d reviews, vocabulary %d, %d empty after cleaning", report["rows"], report["vocabulary_size"],
                report["empty_after_cleaning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

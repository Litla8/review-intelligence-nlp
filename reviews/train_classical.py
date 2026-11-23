"""Step 5 pipeline: train and compare the classical models, then write the comparison, the best model and charts.

Run from the project root::

    python -m reviews.train_classical --sample    # 40,000 reviews, a few minutes, to check the setup
    python -m reviews.train_classical             # all reviews
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import pandas as pd

from reviews import classical, plots
from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def train_and_evaluate(frame: pd.DataFrame, cfg: ProjectConfig) -> Dict[str, Any]:
    """Run every Step 5 computation on an in-memory frame and return the results (no files are written here)."""
    binary = classical.binary_frame(frame, cfg.labels)
    classes = list(cfg.labels.binary_classes)
    train, val, test = classical.stratified_split(binary, cfg.split)
    results = classical.run_experiments(train, val, test, cfg.classical, classes)
    table, best = results["table"], results["best"]
    best_param = str(table.loc[table["model"] == best, "best_param"].iloc[0])
    pipeline = results["pipelines"][best]
    summary: Dict[str, Any] = {
        "reviews_used": int(len(binary)),
        "split": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "positive_share_test": round(float((test[classical.LABEL] == "positive").mean()), 4),
        "best_model": best, "best_param": best_param,
        "best_test_macro_f1": float(table.loc[table["model"] == best, "test_macro_f1"].iloc[0]),
        "tfidf_baseline_macro_f1_for_step6": float(table.loc[table["features"] == "tfidf", "test_macro_f1"].max()),
        "majority_class_macro_f1": float(table.loc[table["model"] == classical.BASELINE, "test_macro_f1"].iloc[0]),
        "per_class": results["evaluations"][best]["per_class"],
        "table": table.where(table.notna(), None).to_dict(orient="records"),
    }
    if "review_date" in binary.columns:
        summary["chronological_check"] = classical.chronological_check(best, best_param, binary, cfg.classical, cfg.split, classes)
    linear = results["pipelines"]["tfidf_logreg"]  # the interpretable model, even when another one wins on macro-F1
    summary["top_terms"] = classical.top_terms(linear, cfg.classical.top_terms)
    scorer = pipeline if hasattr(pipeline.named_steps["model"], "decision_function") else linear
    return {"summary": summary, "table": table, "pipeline": pipeline, "evaluation": results["evaluations"][best],
            "errors": classical.confident_errors(scorer, test, cfg.classical.error_rows), "classes": classes}


def write_outputs(result: Dict[str, Any], cfg: ProjectConfig, sample: bool) -> Dict[str, Path]:
    """Save the model, tables, JSON summary and charts; sample runs get a ``_sample`` suffix so they never overwrite real ones."""
    c, suffix = cfg.classical, "_sample" if sample else ""
    cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    paths = {"model": cfg.paths.models_path / f"{c.model_name}{suffix}.joblib",
             "comparison": cfg.paths.reports_path / f"{c.comparison_name}{suffix}.csv",
             "results": cfg.paths.reports_path / f"{c.results_name}{suffix}.json",
             "errors": cfg.paths.reports_path / f"{c.errors_name}{suffix}.csv"}
    joblib.dump(result["pipeline"], paths["model"])
    result["table"].to_csv(paths["comparison"], index=False)
    paths["results"].write_text(json.dumps({"sample": sample, **result["summary"]}, indent=2), encoding="utf-8")
    result["errors"].to_csv(paths["errors"], index=False)
    figures = cfg.paths.figures_path
    evaluation = result["evaluation"]
    plots.save_figure(plots.plot_confusion_matrix(evaluation["confusion_matrix"], result["classes"],
                                                  f"{result['summary']['best_model']} (test set)"),
                      figures / f"confusion_matrix_classical{suffix}.png", cfg.eda.dpi)
    plots.save_figure(plots.plot_model_comparison(result["table"]), figures / f"model_comparison_classical{suffix}.png", cfg.eda.dpi)
    return paths


def run(cfg: ProjectConfig, sample: bool = False) -> Dict[str, Any]:
    """Read the Step 3 output, train, and write everything.

    Raises:
        FileNotFoundError: If Step 3 has not been run on the full data.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    frame = pd.read_parquet(source)
    if sample:
        frame = frame.sample(n=min(cfg.classical.sample_rows, len(frame)), random_state=classical.SEED)
    LOGGER.info("training on %d reviews", len(frame))
    result = train_and_evaluate(frame, cfg)
    paths = write_outputs(result, cfg, sample)
    LOGGER.info("best model: %s (test macro-F1 %.4f); wrote %s", result["summary"]["best_model"],
                result["summary"]["best_test_macro_f1"], paths["results"])
    return result["summary"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Classical text classifiers (Step 5).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="a 40,000-review smoke test, written to separate files")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=args.root))
    run(cfg, sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

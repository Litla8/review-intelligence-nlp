"""Step 7 pipeline: LDA and BERTopic on a seeded sample of reviews, with coherence, tables, charts and assignments.

Run from the project root::

    python -m reviews.run_topics --sample               # 15,000 reviews, a few minutes
    python -m reviews.run_topics                        # 100,000 reviews, all sentiments
    python -m reviews.run_topics --scope negative       # only negative reviews: the pain-point view
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd

from reviews import plots, topics
from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)

LdaFn = Callable[[Sequence[Sequence[str]], Any], topics.TopicRun]
BertopicFn = Callable[[pd.DataFrame, Sequence[Sequence[str]], Any, Sequence[str]], topics.TopicRun]


def _suffix(scope: str, sample: bool) -> str:
    return ("" if scope == "all" else f"_{scope}") + ("_sample" if sample else "")


def summarise(frame: pd.DataFrame, runs: Sequence[topics.TopicRun], cfg: ProjectConfig, scope: str, sample: bool) -> Dict[str, Any]:
    """Coherence, diversity, size and target check for every model, as JSON-ready numbers."""
    target = cfg.targets.topic_coherence
    out: Dict[str, Any] = {"sample": sample, "scope": scope, "reviews_used": int(len(frame)), "target_c_v": target}
    for run in runs:
        out[run.name] = {"n_topics": run.n_topics, "c_v": run.coherence, "meets_target": bool(run.coherence > target),
                         "diversity": topics.topic_diversity(list(run.topic_words.values())),
                         **{k: v for k, v in run.extras.items()}}
    return out


def write_outputs(frame: pd.DataFrame, runs: Sequence[topics.TopicRun], summary: Dict[str, Any], cfg: ProjectConfig,
                  scope: str, sample: bool) -> Dict[str, Path]:
    """Save tables, summary, assignments, models and charts; file names carry the scope and ``_sample``."""
    suffix = _suffix(scope, sample)
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.processed_path.mkdir(parents=True, exist_ok=True)
    cfg.paths.models_path.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {"summary": cfg.paths.reports_path / f"{cfg.topics.summary_name}{suffix}.json"}
    overall = float(frame["sentiment"].astype(str).eq("negative").mean())
    assign = pd.DataFrame({"Id": frame["Id"]}) if "Id" in frame.columns else pd.DataFrame(index=frame.index)
    for run in runs:
        table = topics.topic_table(frame, run.assignments, run.topic_words)
        paths[f"{run.name}_table"] = cfg.paths.reports_path / f"topics_{run.name}{suffix}.csv"
        table.to_csv(paths[f"{run.name}_table"], index=False)
        assign[f"{run.name}_topic"] = run.assignments
        plots.save_figure(plots.plot_topic_negativity(table, overall, f"{run.name}: negative share by topic"),
                          cfg.paths.figures_path / f"{run.name}_topic_negativity{suffix}.png", cfg.eda.dpi)
        if run.name == "lda":
            sweep = run.extras.get("coherence_sweep")
            if sweep:
                plots.save_figure(plots.plot_coherence_sweep(sweep, run.extras["chosen_k"], cfg.targets.topic_coherence),
                                  cfg.paths.figures_path / f"coherence_sweep{suffix}.png", cfg.eda.dpi)
            if run.model is not None:
                import joblib
                joblib.dump(run.model, cfg.paths.models_path / f"lda{suffix}.joblib")
        elif run.model is not None and hasattr(run.model, "save"):
            try:  # saving BERTopic is a convenience; a failure must not lose the finished results
                run.model.save(str(cfg.paths.models_path / f"bertopic{suffix}"), serialization="safetensors",
                               save_ctfidf=True, save_embedding_model=cfg.topics.embedding_model)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("could not save the BERTopic model: %s", exc)
    paths["assignments"] = cfg.paths.processed_path / f"topic_assignments{suffix}.parquet"
    assign.to_parquet(paths["assignments"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return paths


def run(cfg: ProjectConfig, sample: bool = False, scope: str = "all", lda_fn: Optional[LdaFn] = None,
        bertopic_fn: Optional[BertopicFn] = None) -> Dict[str, Any]:
    """Read the Step 3 output, fit both topic models, and write everything.

    Raises:
        FileNotFoundError: If Step 3 has not been run on the full data.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    tcfg = cfg.topics.for_smoke() if sample else cfg.topics
    cfg = replace(cfg, topics=tcfg)
    frame = topics.sample_reviews(pd.read_parquet(source), scope, tcfg.sample_rows)
    drop = cfg.eda.display_stopwords
    tokens = topics.tokenise(frame["text_clean"], drop)
    LOGGER.info("modelling %d reviews (scope: %s)", len(frame), scope)
    lda = (lda_fn or topics.run_lda)(tokens, tcfg)
    bert = (bertopic_fn or topics.run_bertopic)(frame, tokens, tcfg, drop)
    summary = summarise(frame, [lda, bert], cfg, scope, sample)
    paths = write_outputs(frame, [lda, bert], summary, cfg, scope, sample)
    LOGGER.info("LDA %d topics c_v %.3f; BERTopic %d topics c_v %.3f; wrote %s", lda.n_topics, lda.coherence,
                bert.n_topics, bert.coherence, paths["summary"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Topic modelling with LDA and BERTopic (Step 7).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="a 15,000-review smoke test, written to separate files")
    parser.add_argument("--scope", choices=topics.SCOPES, default="all", help="'negative' models only negative reviews")
    parser.add_argument("--workers", type=int, default=None, help="LDA parallel jobs (default 3; use 1 if it hangs)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    tcfg = base.topics if args.workers is None else replace(base.topics, workers=args.workers)
    cfg = replace(base, paths=replace(base.paths, root=args.root), topics=tcfg)
    run(cfg, sample=args.sample, scope=args.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

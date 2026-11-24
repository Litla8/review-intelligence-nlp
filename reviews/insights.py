"""Step 8 pipeline: aspect-based insights and monthly trend with drop flags.

Run from the project root (needs the Step 6 model)::

    python -m reviews.insights --sample    # 20,000 reviews, a couple of minutes
    python -m reviews.insights             # all reviews
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from reviews import aspects as asp
from reviews import plots
from reviews.config import ProjectConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def load_sentence_scorer(cfg: ProjectConfig) -> asp.Scorer:
    """Return ``scorer(sentences) -> P(positive)`` using the fine-tuned DistilBERT saved in Step 6.

    Sentences are sorted by length before batching (far less padding), then put back in their original order.

    Raises:
        FileNotFoundError: If the model folder is missing.
        ImportError: If torch / transformers are missing.
        RuntimeError: If PyTorch cannot see a GPU.
    """
    model_dir = cfg.paths.models_path / cfg.transformer.model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(f"{model_dir} not found. Run `python -m reviews.train_transformer` first.")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Step 8 needs torch and transformers (see the Step 6 install commands).") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU; see the Step 6 install commands.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to("cuda").eval()
    max_length, batch = cfg.aspects.max_length, cfg.aspects.batch_size

    def scorer(sentences: Sequence[str]) -> np.ndarray:
        order = np.argsort([len(s) for s in sentences], kind="stable")
        out = np.zeros(len(sentences), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                enc = tokenizer([sentences[i] for i in idx], truncation=True, max_length=max_length, padding=True,
                                return_tensors="pt").to("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**enc).logits.float()
                out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out

    return scorer


def analyse(reviews: pd.DataFrame, cfg: ProjectConfig, scorer: asp.Scorer) -> Dict[str, Any]:
    """Compute every Step 8 table from an in-memory frame (no files are written here)."""
    a = cfg.aspects
    names = list(a.keywords)
    sentences = asp.split_sentences(reviews, a.min_sentence_words)
    tagged = asp.tag_aspects(sentences, asp.compile_lexicon(a))
    LOGGER.info("%d sentences, %d mention an aspect; scoring them", len(sentences), len(tagged))
    scored = asp.score_sentences(tagged, scorer, a)
    summary = asp.aspect_summary(scored, reviews, names)
    yearly = asp.aspect_trend(scored, names, "Y", a.min_period_sentences * 3)
    monthly_aspects = asp.aspect_trend(scored, names, "M", a.min_period_sentences)
    overall = asp.monthly_overall(reviews, a.min_period_reviews)
    flags = {"share_positive": asp.flag_drops(overall["share_positive"], a.drop_window, a.drop_min_periods, a.drop_z, "down"),
             "mean_score": asp.flag_drops(overall["mean_score"], a.drop_window, a.drop_min_periods, a.drop_z, "down")}
    aspect_flags = {name: asp.flag_drops(monthly_aspects[name].dropna(), a.drop_window, a.drop_min_periods, a.drop_z, "up")
                    for name in names if monthly_aspects[name].notna().sum() >= a.drop_min_periods + 1}
    return {"summary": summary, "yearly": yearly, "monthly_aspects": monthly_aspects, "overall": overall, "flags": flags,
            "aspect_flags": aspect_flags, "examples": asp.top_examples(scored, names, a.examples_per_aspect),
            "counts": {"reviews": int(len(reviews)), "sentences": int(len(sentences)), "aspect_sentences": int(len(tagged)),
                       "confident_share": round(float((scored["polarity"] != 0).mean()), 4) if len(scored) else 0.0}}


def _flag_rows(flags: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Flagged periods as ``{"2012-03": {"value": ..., "baseline": ..., "z": ...}}``."""
    hit = flags[flags["flagged"]]
    return {ts.strftime("%Y-%m"): {"value": float(r["value"]), "baseline": float(r["baseline"]), "z": float(r["z"])}
            for ts, r in hit.iterrows()}


def write_outputs(result: Dict[str, Any], cfg: ProjectConfig, sample: bool) -> Dict[str, Path]:
    """Save tables, JSON summary and charts; sample runs get a ``_sample`` suffix."""
    suffix = "_sample" if sample else ""
    reports, figures, dpi = cfg.paths.reports_path, cfg.paths.figures_path, cfg.eda.dpi
    reports.mkdir(parents=True, exist_ok=True)
    paths = {"aspects": reports / f"aspect_summary{suffix}.csv", "yearly": reports / f"aspect_trend_yearly{suffix}.csv",
             "monthly": reports / f"monthly_trend{suffix}.csv", "examples": reports / f"aspect_examples{suffix}.csv",
             "summary": reports / f"{cfg.aspects.summary_name}{suffix}.json"}
    result["summary"].to_csv(paths["aspects"], index=False)
    result["yearly"].to_csv(paths["yearly"])
    result["overall"].join(result["flags"]["share_positive"][["flagged"]].rename(columns={"flagged": "drop_flag"})).to_csv(paths["monthly"])
    result["examples"].to_csv(paths["examples"], index=False)
    summary = {"sample": sample, **result["counts"], "aspects": result["summary"].to_dict(orient="records"),
               "drop_flags": {"share_positive": _flag_rows(result["flags"]["share_positive"]),
                              "mean_score": _flag_rows(result["flags"]["mean_score"]),
                              "aspect_negativity_rises": {k: _flag_rows(v) for k, v in result["aspect_flags"].items() if v["flagged"].any()}}}
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plots.save_figure(plots.plot_aspect_pain_points(result["summary"]), figures / f"aspect_pain_points{suffix}.png", dpi)
    plots.save_figure(plots.plot_aspect_trend(result["yearly"]), figures / f"aspect_trend{suffix}.png", dpi)
    plots.save_figure(plots.plot_monthly_flags(result["flags"]["share_positive"], "Positive share by month, with drop flags",
                                               "Share of positive reviews"), figures / f"monthly_drop_flags{suffix}.png", dpi)
    return paths


def run(cfg: ProjectConfig, sample: bool = False, scorer: Optional[asp.Scorer] = None) -> Dict[str, Any]:
    """Read the Step 3 output, run the aspect analysis and write everything.

    Raises:
        FileNotFoundError: If Step 3 or Step 6 has not been run.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    reviews = pd.read_parquet(source, columns=["Id", "Text", "Score", "sentiment", "review_date"])
    if sample:
        reviews = reviews.sample(n=min(cfg.aspects.sample_rows, len(reviews)), random_state=42).sort_index()
    reviews = reviews.reset_index(drop=True)
    result = analyse(reviews, cfg, scorer or load_sentence_scorer(cfg))
    paths = write_outputs(result, cfg, sample)
    LOGGER.info("wrote %s", paths["summary"])
    return json.loads(paths["summary"].read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Aspect-based insights and trend flags (Step 8).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="a 20,000-review smoke test, written to separate files")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 9 summaries: what do unhappy customers say about each aspect?

For each aspect, pick the most representative negative reviews that mention it (closest to the group's average
TF-IDF vector), then summarise them with a BART model. Reviews are packed into chunks that fit the model, each chunk
is summarised, and the summaries are summarised again until one remains (map-reduce). The model is injected as a
function, so everything except :func:`load_summarizer` is testable without it.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from reviews import aspects as asp
from reviews.config import ProjectConfig, SummaryConfig
from reviews.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
SEED = 42
SummarizeFn = Callable[[Sequence[str]], List[str]]
CountFn = Callable[[str], int]
_TAGS = re.compile(r"<[^>]+>")


def clean_for_summary(text: object) -> str:
    """Raw review text with HTML tags and repeated spaces removed (summaries read the original wording)."""
    return re.sub(r"\s+", " ", _TAGS.sub(" ", str(text))).strip()


def representative_positions(texts: pd.Series, k: int, min_words: int, max_words: int, cap: int) -> List[int]:
    """Positions of the ``k`` reviews closest to the group's average TF-IDF vector (the most "typical" ones).

    Reviews outside the word-count range are skipped. At most ``cap`` candidates are used (seeded sample), and the
    result is deterministic.
    """
    words = texts.fillna("").astype(str).str.split().str.len()
    candidates = np.flatnonzero(((words >= min_words) & (words <= max_words)).to_numpy())
    if len(candidates) == 0:
        return []
    if len(candidates) > cap:
        candidates = np.sort(np.random.default_rng(SEED).choice(candidates, cap, replace=False))
    docs = texts.iloc[candidates].astype(str)
    matrix = TfidfVectorizer(stop_words="english", min_df=2 if len(docs) > 20 else 1, max_features=20_000,
                             sublinear_tf=True).fit_transform(docs)
    centroid = np.asarray(matrix.mean(axis=0)).ravel()
    scores = matrix @ centroid
    best = np.argsort(-np.asarray(scores).ravel(), kind="stable")[:k]
    return [int(candidates[i]) for i in best]


def pack_chunks(texts: Sequence[str], count_tokens: CountFn, max_tokens: int) -> List[str]:
    """Greedily join texts (in order) into chunks of at most ``max_tokens``; a single over-long text becomes its own chunk."""
    chunks: List[str] = []
    current: List[str] = []
    used = 0
    for text in texts:
        size = count_tokens(text)
        if current and used + size > max_tokens:
            chunks.append(" ".join(current))
            current, used = [], 0
        current.append(text)
        used += size
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_texts(texts: Sequence[str], summarize: SummarizeFn, count_tokens: CountFn,
                    cfg: SummaryConfig = SummaryConfig()) -> Dict[str, Any]:
    """Map-reduce summary of many texts. Returns ``{"summary", "chunks", "levels"}``.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    cleaned = [t for t in (clean_for_summary(x) for x in texts) if t]
    if not cleaned:
        raise ValueError("nothing to summarise")
    first_chunks = chunks = pack_chunks(cleaned, count_tokens, cfg.max_input_tokens)
    levels = 0
    while True:
        outputs: List[str] = []
        for start in range(0, len(chunks), cfg.batch_size):
            outputs.extend(s.strip() for s in summarize(chunks[start:start + cfg.batch_size]))
        levels += 1
        if len(outputs) == 1 or levels >= cfg.max_levels:
            return {"summary": " ".join(outputs), "chunks": len(first_chunks), "levels": levels}
        chunks = pack_chunks(outputs, count_tokens, cfg.max_input_tokens)


def load_summarizer(cfg: SummaryConfig = SummaryConfig()) -> Tuple[SummarizeFn, CountFn]:
    """Load the BART summariser and return ``(summarize, count_tokens)``; uses the GPU when there is one.

    Uses the model classes directly (the "summarization" pipeline was removed from recent transformers versions).

    Raises:
        ImportError: If torch / transformers are missing.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Summaries need torch and transformers (see the Step 6 install commands).") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(device).eval()

    def summarize(chunks: Sequence[str]) -> List[str]:
        enc = tokenizer(list(chunks), truncation=True, max_length=cfg.max_input_tokens + 100, padding=True,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=cfg.num_beams, max_new_tokens=cfg.max_new_tokens,
                                 min_new_tokens=cfg.min_new_tokens, no_repeat_ngram_size=3, early_stopping=True)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return summarize, count_tokens


def summarize_group(reviews: pd.DataFrame, cfg: SummaryConfig, summarize: SummarizeFn, count_tokens: CountFn,
                    text_column: str = "Text", feature_column: str = "text_clean") -> Optional[Dict[str, Any]]:
    """Summary of the most representative reviews in ``reviews``, or None if none are usable."""
    positions = representative_positions(reviews[feature_column], cfg.reviews_per_group, cfg.min_review_words,
                                         cfg.max_review_words, cfg.candidate_cap)
    if not positions:
        return None
    chosen = reviews.iloc[positions]
    result = summarize_texts(chosen[text_column].tolist(), summarize, count_tokens, cfg)
    result["example_ids"] = [int(i) for i in chosen["Id"].head(5)] if "Id" in chosen.columns else []
    result["reviews_summarised"] = int(len(chosen))
    return result


def summarize_pain_points(negative: pd.DataFrame, cfg: ProjectConfig, summarize: SummarizeFn,
                          count_tokens: CountFn) -> Dict[str, Any]:
    """One summary per aspect (negative reviews that mention it) plus one for all negative reviews."""
    sc = cfg.summary
    lexicon = asp.compile_lexicon(cfg.aspects)
    groups: Dict[str, pd.DataFrame] = {"overall": negative}
    for aspect, pattern in lexicon.items():
        groups[aspect] = negative[negative["Text"].astype(str).str.contains(pattern)]
    results: Dict[str, Any] = {}
    for name, group in groups.items():
        if len(group) < sc.min_group_reviews and name != "overall":
            results[name] = {"negative_reviews": int(len(group)), "skipped": f"fewer than {sc.min_group_reviews} negative reviews"}
            continue
        summary = summarize_group(group, sc, summarize, count_tokens)
        results[name] = {"negative_reviews": int(len(group)), **(summary or {"skipped": "no reviews in the usable length range"})}
        LOGGER.info("%s: %d negative reviews summarised", name, len(group))
    return results


def render_markdown(results: Dict[str, Any], model_name: str) -> str:
    """The summaries as a readable Markdown report (with a note that they are machine-written)."""
    lines = ["# Pain-point summaries", "",
             f"Machine-written by `{model_name}` from the most representative negative reviews of each aspect. "
             "Check against the example reviews before quoting.", ""]
    for name, item in results.items():
        lines += [f"## {name.capitalize()} ({item['negative_reviews']:,} negative reviews)", ""]
        lines += [item["summary"] if "summary" in item else f"_Skipped: {item['skipped']}._", ""]
        if item.get("example_ids"):
            lines += [f"Example review Ids: {', '.join(map(str, item['example_ids']))}", ""]
    return "\n".join(lines)


def run(cfg: ProjectConfig, sample: bool = False,
        loader: Optional[Callable[[SummaryConfig], Tuple[SummarizeFn, CountFn]]] = None) -> Dict[str, Any]:
    """Read the Step 3 output, summarise the negative reviews per aspect, and write JSON and Markdown.

    Raises:
        FileNotFoundError: If Step 3 has not been run.
    """
    source = cfg.paths.text_path(sample=False)
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found. Run `python -m reviews.preprocess` first.")
    frame = pd.read_parquet(source, columns=["Id", "Text", "text_clean", "sentiment"])
    negative = frame[frame["sentiment"].astype(str) == "negative"].reset_index(drop=True)
    if sample:
        negative = negative.sample(n=min(cfg.summary.sample_rows, len(negative)), random_state=SEED).reset_index(drop=True)
        cfg = replace(cfg, summary=replace(cfg.summary, min_group_reviews=5, reviews_per_group=8))
    summarize, count_tokens = (loader or load_summarizer)(cfg.summary)
    results = summarize_pain_points(negative, cfg, summarize, count_tokens)
    suffix = "_sample" if sample else ""
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_path / f"{cfg.summary.report_name}{suffix}.json").write_text(
        json.dumps({"sample": sample, "model": cfg.summary.model_name, "groups": results}, indent=2), encoding="utf-8")
    (cfg.paths.reports_path / f"{cfg.summary.report_name}{suffix}.md").write_text(
        render_markdown(results, cfg.summary.model_name), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Summarise negative reviews per aspect (Step 9).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--sample", action="store_true", help="3,000 negative reviews, written to separate files")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING ...")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    run(replace(base, paths=replace(base.paths, root=args.root)), sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

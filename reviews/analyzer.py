"""What the Streamlit app computes: one pasted review, or an uploaded file of reviews.

Everything takes the scoring functions as arguments (see ``reviews/predict.py``), so it is testable without a GPU
and the app file only has to draw the results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from reviews import aspects as asp
from reviews.config import AspectConfig, AppConfig, LabelConfig
from reviews.labels import stars_to_sentiment
from reviews.metrics import macro_f1
from reviews.predict import Scorer, label_probabilities

LOGGER = logging.getLogger(__name__)
SEED = 42
TEXT_GUESSES = ("text", "review", "reviewtext", "review_text", "content", "comment", "body")
RATING_GUESSES = ("score", "rating", "stars", "star_rating")
DATE_GUESSES = ("time", "date", "review_date", "timestamp", "created", "created_at")


@dataclass
class Scorers:
    """The two scoring functions the analysis needs, plus the tuned threshold."""

    review: Scorer
    sentence: Scorer
    threshold: float = 0.5


@dataclass
class FrameAnalysis:
    """Results for an uploaded file."""

    reviews: pd.DataFrame
    aspects: pd.DataFrame
    examples: pd.DataFrame
    monthly: Optional[pd.DataFrame] = None
    flags: Optional[pd.DataFrame] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def guess_column(columns: List[str], guesses: tuple) -> Optional[str]:
    """The first column whose lower-cased name is in ``guesses`` (to pre-select dropdowns), or None."""
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[g] for g in guesses if g in lookup), None)


def parse_dates(values: pd.Series) -> pd.Series:
    """Dates from text or from Unix seconds (the Amazon file's format); unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.9 and (numeric.dropna() > 1e8).all():
        return pd.to_datetime(numeric, unit="s", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def analyze_review(text: str, scorers: Scorers, cfg: AspectConfig = AspectConfig(), min_words: int = 2) -> Dict[str, Any]:
    """Sentiment of one review and the sentiment of each sentence that mentions an aspect.

    Raises:
        ValueError: If ``text`` is empty.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Paste a review first")
    p = float(scorers.review([text])[0])
    label = str(label_probabilities(np.array([p]), scorers.threshold, cfg)[0])
    one = pd.DataFrame({"Text": [text], "review_date": pd.to_datetime([None]), "sentiment": [label]})
    tagged = asp.tag_aspects(asp.split_sentences(one, min_words), asp.compile_lexicon(cfg))
    scored = asp.score_sentences(tagged, scorers.sentence, cfg)
    names = list(cfg.keywords)
    rows = [{"sentence": r["sentence"], "aspects": ", ".join(a for a in names if r[a]),
             "p_positive": round(float(r["p_positive"]), 3),
             "sentiment": {1: "positive", 0: "mixed", -1: "negative"}[int(r["polarity"])]}
            for _, r in scored.iterrows()]
    return {"label": label, "p_positive": round(p, 4), "confident": bool(label != "mixed"),
            "words": len(text.split()), "aspect_sentences": rows}


def analyze_frame(df: pd.DataFrame, text_column: str, scorers: Scorers, app: AppConfig = AppConfig(),
                  cfg: AspectConfig = AspectConfig(), labels: LabelConfig = LabelConfig(),
                  rating_column: Optional[str] = None, date_column: Optional[str] = None) -> FrameAnalysis:
    """Score every review, then compute aspect pain points, a monthly trend with drop flags, and example quotes.

    With a rating column the reviewer's own stars define positive / negative (the model's view is reported as
    agreement); without one the model's label is used. A seeded sample is taken above ``app.max_rows`` reviews.

    Raises:
        ValueError: If a named column is missing or no review has text.
    """
    for name in (text_column, rating_column, date_column):
        if name is not None and name not in df.columns:
            raise ValueError(f"Column {name!r} not found")
    frame = pd.DataFrame({"Text": df[text_column].fillna("").astype(str).str.strip()})
    frame["review_date"] = parse_dates(df[date_column]) if date_column else pd.NaT
    frame["review_date"] = pd.to_datetime(frame["review_date"])
    if rating_column:
        frame["Score"] = pd.to_numeric(df[rating_column], errors="coerce")
    frame = frame[frame["Text"].str.len() > 0]
    if rating_column:
        frame = frame[frame["Score"].between(1, 5)]
    if frame.empty:
        raise ValueError("No reviews with text were found in that column")
    total = len(frame)
    if total > app.max_rows:
        frame = frame.sample(n=app.max_rows, random_state=SEED).sort_index()
    frame = frame.reset_index(drop=True)

    frame["p_positive"] = np.asarray(scorers.review(frame["Text"].tolist()), dtype=float)
    frame["model_sentiment"] = label_probabilities(frame["p_positive"].to_numpy(), scorers.threshold, cfg)
    frame["model_sentiment"] = frame["model_sentiment"].replace("mixed", "neutral")
    frame["sentiment"] = (stars_to_sentiment(frame["Score"], labels).astype(str) if rating_column
                          else frame["model_sentiment"])

    tagged = asp.tag_aspects(asp.split_sentences(frame, cfg.min_sentence_words), asp.compile_lexicon(cfg))
    scored = asp.score_sentences(tagged, scorers.sentence, cfg)
    names = list(cfg.keywords)
    result = FrameAnalysis(frame, asp.aspect_summary(scored, frame, names), asp.top_examples(scored, names, app.quotes_per_aspect))
    result.metrics = {"reviews_analysed": int(len(frame)), "reviews_in_file": int(total),
                      "negative_share": round(float((frame["sentiment"] == "negative").mean()), 4),
                      "positive_share": round(float((frame["sentiment"] == "positive").mean()), 4),
                      "mean_p_positive": round(float(frame["p_positive"].mean()), 4), "sampled": bool(total > app.max_rows)}
    if rating_column:
        both = frame[(frame["sentiment"] != "neutral") & (frame["model_sentiment"] != "neutral")]
        result.metrics["model_vs_stars_macro_f1"] = round(macro_f1(both["sentiment"], both["model_sentiment"]), 4) if len(both) else None
        result.metrics["model_vs_stars_coverage"] = round(len(both) / len(frame), 4)
    if frame["review_date"].notna().any():
        dated = frame.dropna(subset=["review_date"])
        monthly = asp.monthly_overall(dated, cfg.min_period_reviews)
        if len(monthly) >= cfg.drop_min_periods + 1:
            result.monthly = monthly
            result.flags = asp.flag_drops(monthly["share_positive"], cfg.drop_window, cfg.drop_min_periods, cfg.drop_z, "down")
    return result

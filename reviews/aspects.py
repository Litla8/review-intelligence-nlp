"""Step 8 analysis: aspect mentions, sentence-level sentiment per aspect, trends and drop flags.

Reviews are split into sentences; a sentence that contains an aspect keyword is scored by the fine-tuned DistilBERT
(injected as a function, so everything here is testable without a GPU). Counting mentions in negative reviews gives
the pain-point ranking; sentence sentiment says whether people are positive or negative *about that aspect*.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reviews.config import AspectConfig

LOGGER = logging.getLogger(__name__)
_SENTENCE_BREAK = r"(?<=[.!?])\s+"
_TAGS = re.compile(r"<[^>]+>")
Scorer = Callable[[Sequence[str]], np.ndarray]


def compile_lexicon(cfg: AspectConfig = AspectConfig()) -> Dict[str, "re.Pattern[str]"]:
    """One case-insensitive whole-word regex per aspect (so "late" does not match "chocolate")."""
    return {aspect: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
            for aspect, words in cfg.keywords.items()}


def split_sentences(frame: pd.DataFrame, min_words: int = 3) -> pd.DataFrame:
    """One row per sentence with its review's position, id, date and sentiment; HTML tags are removed.

    Sentences shorter than ``min_words`` words ("Great!") carry no aspect and are dropped.
    """
    missing = [c for c in ("Text", "review_date", "sentiment") if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    text = frame["Text"].fillna("").astype(str).str.replace(_TAGS, " ", regex=True)
    pieces = text.str.split(_SENTENCE_BREAK, regex=True)
    out = pd.DataFrame({"review_pos": np.arange(len(frame)), "review_date": frame["review_date"].to_numpy(),
                        "review_sentiment": frame["sentiment"].astype(str).to_numpy(), "sentence": pieces.to_numpy()})
    if "Id" in frame.columns:
        out["Id"] = frame["Id"].to_numpy()
    out = out.explode("sentence", ignore_index=True)
    out["sentence"] = out["sentence"].astype(str).str.strip()
    return out[out["sentence"].str.split().str.len().ge(min_words)].reset_index(drop=True)


def tag_aspects(sentences: pd.DataFrame, lexicon: Mapping[str, "re.Pattern[str]"]) -> pd.DataFrame:
    """Add one boolean column per aspect and keep only sentences that mention at least one."""
    tagged = sentences.copy()
    for aspect, pattern in lexicon.items():
        tagged[aspect] = tagged["sentence"].str.contains(pattern)
    return tagged[tagged[list(lexicon)].any(axis=1)].reset_index(drop=True)


def polarity(p_positive: np.ndarray, cfg: AspectConfig = AspectConfig()) -> np.ndarray:
    """-1 (negative), 0 (neutral or mixed: probability between the two bands) or +1 (positive) for each sentence.

    A binary classifier must pick a side even for "It arrived on Tuesday"; the band keeps those out of the shares.
    """
    p = np.asarray(p_positive, dtype=float)
    return np.where(p >= cfg.neutral_high, 1, np.where(p <= cfg.neutral_low, -1, 0)).astype(np.int8)


def score_sentences(tagged: pd.DataFrame, scorer: Scorer, cfg: AspectConfig = AspectConfig()) -> pd.DataFrame:
    """Add ``p_positive`` and ``polarity`` columns by running ``scorer`` on every tagged sentence."""
    if tagged.empty:
        return tagged.assign(p_positive=np.array([], dtype=float), polarity=np.array([], dtype=np.int8))
    p = np.asarray(scorer(tagged["sentence"].tolist()), dtype=float)
    if p.shape != (len(tagged),):
        raise ValueError("scorer must return one probability per sentence")
    return tagged.assign(p_positive=p, polarity=polarity(p, cfg))


def aspect_summary(scored: pd.DataFrame, reviews: pd.DataFrame, aspects: Sequence[str]) -> pd.DataFrame:
    """One row per aspect, ranked by how many negative reviews raise it.

    * ``neg_review_rate`` / ``pos_review_rate``: share of all negative / positive reviews that mention the aspect;
      ``pain_ratio`` is their ratio (above 1: negative reviewers bring it up more).
    * ``neg_sentence_share``: of the sentences about the aspect that the model is confident about, the share that are negative.
    * ``net_sentiment``: positive minus negative share among those sentences.
    """
    sentiment = reviews["sentiment"].astype(str).to_numpy()
    n_neg, n_pos = int((sentiment == "negative").sum()), int((sentiment == "positive").sum())
    rows = []
    for aspect in aspects:
        hit = scored[scored[aspect]]
        reviews_hit = hit.drop_duplicates("review_pos")
        by_sent = reviews_hit["review_sentiment"].value_counts()
        polar = hit[hit["polarity"] != 0]
        neg_rate = by_sent.get("negative", 0) / n_neg if n_neg else np.nan
        pos_rate = by_sent.get("positive", 0) / n_pos if n_pos else np.nan
        rows.append({"aspect": aspect, "reviews": int(len(reviews_hit)), "mention_rate": round(len(reviews_hit) / len(reviews), 4),
                     "neg_review_rate": round(neg_rate, 4), "pos_review_rate": round(pos_rate, 4),
                     "pain_ratio": round(neg_rate / pos_rate, 2) if pos_rate else np.nan,
                     "sentences": int(len(hit)), "polar_sentences": int(len(polar)),
                     "neg_sentence_share": round(float((polar["polarity"] == -1).mean()), 4) if len(polar) else np.nan,
                     "net_sentiment": round(float(polar["polarity"].mean()), 4) if len(polar) else np.nan})
    return pd.DataFrame(rows).sort_values("neg_review_rate", ascending=False, kind="stable").reset_index(drop=True)


def aspect_trend(scored: pd.DataFrame, aspects: Sequence[str], freq: str, min_sentences: int) -> pd.DataFrame:
    """Negative share of confident sentences per aspect per period (``"Y"`` or ``"M"``); thin periods are left blank."""
    polar = scored[scored["polarity"] != 0]
    period = polar["review_date"].dt.to_period(freq).dt.to_timestamp()
    columns = {}
    for aspect in aspects:
        mask = polar[aspect].to_numpy()
        grouped = (polar.loc[mask, "polarity"] == -1).groupby(period[mask])
        share, count = grouped.mean(), grouped.size()
        columns[aspect] = share.where(count >= min_sentences).round(4)
    trend = pd.DataFrame(columns).sort_index()
    trend.index.name = "period"
    return trend


def flag_drops(series: pd.Series, window: int, min_periods: int, z: float, direction: str = "down") -> pd.DataFrame:
    """Flag periods that are ``z`` standard deviations worse than the average of the previous ``window`` periods.

    ``direction="down"`` flags falls (for positive share), ``"up"`` flags rises (for negative share). The baseline uses
    only earlier periods (shifted by one), so a flagged month cannot hide inside its own baseline.
    """
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'")
    past = series.shift(1).rolling(window, min_periods=min_periods)
    mean, std = past.mean(), past.std()
    score = (series - mean) / std.where(std > 0)
    flagged = score <= -z if direction == "down" else score >= z
    out = pd.DataFrame({"value": series, "baseline": mean.round(4), "z": score.round(2), "flagged": flagged.fillna(False)})
    return out


def monthly_overall(reviews: pd.DataFrame, min_reviews: int) -> pd.DataFrame:
    """Reviews, mean stars and positive / negative share per month (months with few reviews dropped)."""
    month = reviews["review_date"].dt.to_period("M").dt.to_timestamp()
    sentiment = reviews["sentiment"].astype(str)
    score = reviews["Score"] if "Score" in reviews.columns else pd.Series(np.nan, index=reviews.index)  # uploads may have no stars
    frame = pd.DataFrame({"n_reviews": month.value_counts().sort_index(), "mean_score": score.groupby(month).mean(),
                          "share_positive": (sentiment == "positive").groupby(month).mean(),
                          "share_negative": (sentiment == "negative").groupby(month).mean()})
    frame.index.name = "month"
    return frame[frame["n_reviews"] >= min_reviews].round(4)


def top_examples(scored: pd.DataFrame, aspects: Sequence[str], per_aspect: int) -> pd.DataFrame:
    """The most confidently negative sentences for each aspect (quotes for the pain-point report)."""
    parts: List[pd.DataFrame] = []
    for aspect in aspects:
        hit = scored[scored[aspect] & (scored["polarity"] == -1)].sort_values("p_positive", kind="stable").head(per_aspect)
        parts.append(hit[["sentence", "p_positive"]].assign(aspect=aspect))
    cols = ["aspect", "p_positive", "sentence"]
    return pd.concat(parts)[cols].reset_index(drop=True) if parts else pd.DataFrame(columns=cols)

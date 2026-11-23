"""Load the raw review file and clean it, one small function per rule.

Every cleaning function takes a frame and returns the kept rows, so each rule can be tested alone and the
pipeline can report exactly how many rows each rule removed. Review text is never modified here: the raw text
stays available for the transformer models in Step 6.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from reviews.config import ENGLISH_STOPWORDS, DataConfig, LabelConfig
from reviews.labels import stars_to_sentiment

LOGGER = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_NON_ASCII = r"[^\x00-\x7f]"
_STOPWORD_PATTERN = r"\b(?:" + "|".join(ENGLISH_STOPWORDS) + r")\b"


def load_raw(path: Path, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Read the Kaggle ``Reviews.csv`` with compact dtypes.

    Raises:
        FileNotFoundError: If the file is missing, with the place to download it.
        ValueError: If a required column is absent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download 'Amazon Fine Food Reviews' from Kaggle and put "
                                f"Reviews.csv in data/raw/.")
    dtypes = {cfg.helpful_yes_column: "float64", cfg.helpful_total_column: "float64"}
    frame = pd.read_csv(path, dtype=dtypes)
    validate_columns(frame, cfg)
    return frame


def validate_columns(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> None:
    """Raise if any column the pipeline needs is missing."""
    missing = [c for c in cfg.required_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Reviews file is missing columns: {missing}")


def coerce_types(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Numeric rating, datetime ``review_date`` from unix seconds; unparseable values become NaN / NaT."""
    out = frame.copy()
    out[cfg.rating_column] = pd.to_numeric(out[cfg.rating_column], errors="coerce")
    out["review_date"] = pd.to_datetime(pd.to_numeric(out[cfg.time_column], errors="coerce"), unit="s", errors="coerce")
    return out


def drop_unusable_text(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Drop reviews whose text is missing, blank or shorter than ``cfg.min_words`` words."""
    text = frame[cfg.text_column]
    words = text.fillna("").astype(str).str.split().str.len()
    return frame[text.notna() & (words >= cfg.min_words)]


def drop_invalid_ratings(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Keep only whole-number ratings from 1 to 5 (a review without a rating has no label)."""
    rating = frame[cfg.rating_column]
    valid = rating.between(1, 5) & (rating == rating.round())
    return frame[valid & frame["review_date"].notna()]


def drop_duplicate_reviews(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Drop exact duplicates by ``cfg.duplicate_key``, keeping the first row."""
    return frame.drop_duplicates(subset=list(cfg.duplicate_key), keep="first")


def normalise_text(text: pd.Series) -> pd.Series:
    """Lower-case and collapse whitespace, used only to find copies (the stored text is untouched)."""
    return text.astype(str).str.lower().str.replace(_WHITESPACE, " ", regex=True).str.strip()


def drop_duplicate_texts(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Drop later copies of the same words, even under another product or user.

    The same review is often posted once for every size or flavour of a product. If copies sat in both the
    training and the test set, the test score would be inflated, so only the first copy is kept.
    """
    if not cfg.drop_duplicate_texts:
        return frame
    return frame[~normalise_text(frame[cfg.text_column]).duplicated(keep="first")]


def english_mask(text: pd.Series, cfg: DataConfig = DataConfig()) -> pd.Series:
    """True where the text looks English: mostly ASCII, and (for longer texts) several common English words."""
    lowered = text.fillna("").astype(str).str.lower()
    length = lowered.str.len().clip(lower=1)
    ascii_share = 1.0 - lowered.str.count(_NON_ASCII) / length
    words = lowered.str.split().str.len().fillna(0)
    stopword_hits = lowered.str.count(_STOPWORD_PATTERN)
    long_enough = words >= cfg.short_text_words
    return (ascii_share >= cfg.min_ascii_share) & (~long_enough | (stopword_hits >= cfg.min_stopword_hits))


def drop_non_english(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> pd.DataFrame:
    """Drop reviews that do not look English (skipped if ``cfg.english_only`` is False)."""
    if not cfg.english_only:
        return frame
    return frame[english_mask(frame[cfg.text_column], cfg)]


def add_helpfulness_ratio(frame: pd.DataFrame, cfg: DataConfig = DataConfig()) -> Tuple[pd.DataFrame, int]:
    """Add ``helpful_ratio`` (yes votes / total votes). Impossible vote counts give NaN; returns how many."""
    yes, total = frame[cfg.helpful_yes_column], frame[cfg.helpful_total_column]
    invalid = (yes > total) | (yes < 0) | (total < 0)
    ratio = (yes / total.where(total > 0)).where(~invalid)
    return frame.assign(helpful_ratio=ratio), int(invalid.sum())


def add_labels(frame: pd.DataFrame, cfg: DataConfig = DataConfig(), labels: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """Add the three-class ``sentiment`` label and the review length in words."""
    words = frame[cfg.text_column].astype(str).str.split().str.len().astype("int32")
    return frame.assign(sentiment=stars_to_sentiment(frame[cfg.rating_column], labels), n_words=words)


CleaningStep = Callable[[pd.DataFrame, DataConfig], pd.DataFrame]

# Order matters: cheap row filters first, then duplicates, then the language check on what is left.
CLEANING_STEPS: Tuple[Tuple[str, CleaningStep], ...] = (
    ("unusable_text", drop_unusable_text),
    ("invalid_rating_or_date", drop_invalid_ratings),
    ("duplicate_review", drop_duplicate_reviews),
    ("duplicate_text", drop_duplicate_texts),
    ("non_english", drop_non_english),
)


@dataclass(frozen=True)
class CleaningResult:
    """The cleaned frame plus what each rule removed."""

    frame: pd.DataFrame
    stages: List[Dict[str, int]]
    invalid_helpfulness_votes: int


def clean_reviews(raw: pd.DataFrame, cfg: DataConfig = DataConfig(),
                  labels: LabelConfig = LabelConfig()) -> CleaningResult:
    """Run every cleaning rule in order, then label the survivors.

    Raises:
        ValueError: If columns are missing or no review survives cleaning.
    """
    validate_columns(raw, cfg)
    frame = coerce_types(raw, cfg)
    stages: List[Dict[str, int]] = []
    for name, step in CLEANING_STEPS:
        before = len(frame)
        frame = step(frame, cfg)
        stages.append({"step": name, "dropped": before - len(frame), "remaining": len(frame)})
        LOGGER.info("%-24s dropped %8d, %8d left", name, before - len(frame), len(frame))
    if frame.empty:
        raise ValueError("No reviews survived cleaning; check the cleaning settings")
    frame, invalid_votes = add_helpfulness_ratio(frame, cfg)
    frame = add_labels(frame, cfg, labels)
    keep = [cfg.id_column, cfg.product_column, cfg.user_column, "review_date", cfg.rating_column, "sentiment",
            cfg.summary_column, cfg.text_column, "n_words", "helpful_ratio"]
    return CleaningResult(frame[keep].reset_index(drop=True), stages, invalid_votes)

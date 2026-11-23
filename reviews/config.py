"""All settings for the review intelligence project in one place (no module-level globals elsewhere)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

SEED = 42


@dataclass(frozen=True)
class PathsConfig:
    """Where files live. Everything is relative to ``root`` so the repo can move."""

    root: Path = Path(".")
    raw_file: str = "data/raw/Reviews.csv"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"

    @property
    def raw_path(self) -> Path:
        return self.root / self.raw_file

    @property
    def processed_path(self) -> Path:
        return self.root / self.processed_dir

    @property
    def models_path(self) -> Path:
        return self.root / self.models_dir

    @property
    def reports_path(self) -> Path:
        return self.root / self.reports_dir


@dataclass(frozen=True)
class DataConfig:
    """Columns of the Kaggle Amazon Fine Food Reviews file and cleaning rules."""

    text_column: str = "Text"
    summary_column: str = "Summary"
    rating_column: str = "Score"
    time_column: str = "Time"              # unix seconds
    product_column: str = "ProductId"
    user_column: str = "UserId"
    helpful_yes_column: str = "HelpfulnessNumerator"
    helpful_total_column: str = "HelpfulnessDenominator"
    min_words: int = 3                      # shorter reviews carry no usable signal
    english_only: bool = True
    drop_duplicates_on: Tuple[str, ...] = ("UserId", "ProductId", "Time", "Text")


@dataclass(frozen=True)
class LabelConfig:
    """Star rating to sentiment, exactly as the brief defines it."""

    negative_max: int = 2                   # 1-2 stars
    positive_min: int = 4                   # 4-5 stars; 3 stars is neutral
    classes: Tuple[str, ...] = ("negative", "neutral", "positive")
    binary_classes: Tuple[str, ...] = ("negative", "positive")


@dataclass(frozen=True)
class SplitConfig:
    """Stratified train/validation/test split (sentiment) plus a time-ordered holdout for drift checks."""

    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = SEED
    stratify: bool = True


@dataclass(frozen=True)
class TargetConfig:
    """Success metrics from the brief, plus the business-facing ones we add."""

    macro_f1_binary: float = 0.88           # transformer, positive vs negative
    topic_coherence: float = 0.50           # c_v, LDA and BERTopic
    min_gain_over_baseline: float = 0.01    # a transformer must beat TF-IDF by this macro-F1 to be worth its cost


def _default_aspects() -> Dict[str, Tuple[str, ...]]:
    """Seed keywords per aspect for food reviews; Step 8 extends them with discovered topics."""
    return {
        "taste": ("taste", "tasty", "flavor", "delicious", "bland", "sweet", "bitter", "salty", "aroma"),
        "price": ("price", "cheap", "expensive", "cost", "value", "worth", "overpriced", "deal"),
        "shipping": ("shipping", "shipped", "delivery", "arrived", "late", "delay", "package"),
        "packaging": ("packaging", "packed", "box", "damaged", "broken", "leak", "seal", "crushed"),
        "freshness": ("fresh", "stale", "expired", "expiration", "rancid", "moldy", "date"),
        "quality": ("quality", "ingredients", "artificial", "organic", "natural", "texture"),
        "health": ("healthy", "sugar", "calories", "allergy", "gluten", "diet", "protein"),
        "service": ("customer service", "refund", "return", "seller", "support", "replacement"),
    }


@dataclass(frozen=True)
class AspectConfig:
    """Aspects for aspect-based sentiment (Step 8)."""

    keywords: Dict[str, Tuple[str, ...]] = field(default_factory=_default_aspects)


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level bundle passed around instead of globals."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    aspects: AspectConfig = field(default_factory=AspectConfig)

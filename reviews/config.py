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

    @property
    def clean_path(self) -> Path:
        """Cleaned, labelled reviews written by Step 2."""
        return self.processed_path / "reviews_clean.parquet"

    def text_path(self, sample: bool = False) -> Path:
        """Preprocessed reviews written by Step 3 (the sample file is for quick checks)."""
        return self.processed_path / (TextConfig.sample_output_name if sample else TextConfig.output_name)

    @property
    def text_report_path(self) -> Path:
        """JSON report written by Step 3."""
        return self.reports_path / TextConfig.report_name

    @property
    def quality_report_path(self) -> Path:
        """JSON data-quality report written by Step 2."""
        return self.reports_path / "data_quality.json"


# Very common English words. A real review of a few sentences contains several of them; text in another language
# or random characters contains almost none. Used by the dependency-free language check in Step 2.
ENGLISH_STOPWORDS: Tuple[str, ...] = (
    "the", "and", "is", "it", "to", "of", "a", "i", "this", "for", "in", "was", "with", "not", "but", "have",
    "are", "that", "my", "you", "on", "very", "so", "as", "they", "be", "at", "were", "or", "we", "had", "has",
)


@dataclass(frozen=True)
class DataConfig:
    """Columns of the Kaggle Amazon Fine Food Reviews file and the cleaning rules."""

    text_column: str = "Text"
    summary_column: str = "Summary"
    rating_column: str = "Score"
    time_column: str = "Time"              # unix seconds
    product_column: str = "ProductId"
    user_column: str = "UserId"
    helpful_yes_column: str = "HelpfulnessNumerator"
    helpful_total_column: str = "HelpfulnessDenominator"
    id_column: str = "Id"
    min_words: int = 3                      # shorter reviews carry no usable signal
    english_only: bool = True
    # Exact duplicates: the same person posting the same words for the same product at the same moment.
    duplicate_key: Tuple[str, ...] = ("UserId", "ProductId", "Time", "Text")
    drop_duplicate_texts: bool = True       # the same review copied across product variants; leaks across splits
    # Language check (no extra dependency): mostly ASCII letters, and enough common English words.
    min_ascii_share: float = 0.9
    short_text_words: int = 6               # below this many words only the ASCII test applies
    min_stopword_hits: int = 2

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return (self.id_column, self.product_column, self.user_column, self.helpful_yes_column,
                self.helpful_total_column, self.rating_column, self.time_column, self.summary_column,
                self.text_column)

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.min_words < 1 or self.short_text_words < 1 or self.min_stopword_hits < 0:
            raise ValueError("word thresholds must be positive")
        if not 0 <= self.min_ascii_share <= 1:
            raise ValueError("min_ascii_share must be between 0 and 1")
        if not self.duplicate_key:
            raise ValueError("duplicate_key must not be empty")


# Words kept even though they appear in the usual stop-word lists: they flip or scale sentiment
# ("not good" must not become "good").
SENTIMENT_WORDS: Tuple[str, ...] = (
    "not", "no", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot", "without",
    "but", "very", "too", "more", "most", "less", "least", "few", "only", "against", "hardly", "barely",
)


@dataclass(frozen=True)
class TextConfig:
    """Step 3 text preprocessing settings."""

    spacy_model: str = "en_core_web_sm"
    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    map_emoticons: bool = True              # ":)" and ":(" carry sentiment in 2007-2012 reviews
    expand_contractions: bool = True
    remove_digits: bool = True
    remove_stopwords: bool = True
    keep_words: Tuple[str, ...] = SENTIMENT_WORDS
    min_token_length: int = 2               # drops stray single letters left by cleaning
    batch_size: int = 1000
    n_process: int = 4                      # spaCy worker processes; 1 is the safe fallback
    sample_rows: int = 5000                 # size of the quick smoke-test run
    output_name: str = "reviews_text.parquet"
    sample_output_name: str = "reviews_text_sample.parquet"
    report_name: str = "text_report.json"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if self.batch_size < 1 or self.n_process < 1 or self.sample_rows < 1 or self.min_token_length < 1:
            raise ValueError("batch_size, n_process, sample_rows and min_token_length must be positive")


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
    text: TextConfig = field(default_factory=TextConfig)

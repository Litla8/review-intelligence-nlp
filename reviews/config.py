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
    def figures_path(self) -> Path:
        """Charts written by Step 4."""
        return self.reports_path / EdaConfig.figures_dir

    @property
    def eda_summary_path(self) -> Path:
        """JSON summary written by Step 4."""
        return self.reports_path / EdaConfig.summary_name

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


# Words that are correct for the classifiers but drown out everything else in word clouds, frequency charts and topic
# models: negations and intensifiers, plus verbs and fillers that appear in every review whatever the topic.
DISPLAY_STOPWORDS: Tuple[str, ...] = SENTIMENT_WORDS + (
    "product", "make", "use", "try", "buy", "just", "get", "one", "would", "also", "really", "go", "even", "much",
    "can", "will", "come", "take", "give", "want", "need", "think", "say", "see", "know", "look", "order", "bought",
)


@dataclass(frozen=True)
class EdaConfig:
    """Step 4 exploratory analysis settings."""

    top_n: int = 20                         # words / phrases shown per chart
    ngram_top: int = 15
    ngram_max_docs: int = 150_000           # per sentiment; phrase counting on every review adds time, not insight
    min_count_distinctive: int = 50         # ignore words rarer than this when ranking by distinctiveness
    prior_strength: float = 500.0           # total weight of the smoothing prior in the log-odds z-score
    cloud_max_words: int = 150
    min_month_reviews: int = 200            # months with fewer reviews give noisy shares and are left out of the trend
    length_clip_quantile: float = 0.99      # histogram range, so a few 3,000-word reviews do not flatten the chart
    display_stopwords: Tuple[str, ...] = DISPLAY_STOPWORDS
    dpi: int = 150
    figures_dir: str = "figures"
    summary_name: str = "eda_summary.json"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if min(self.top_n, self.ngram_top, self.ngram_max_docs, self.min_count_distinctive, self.cloud_max_words,
               self.min_month_reviews, self.dpi) < 1:
            raise ValueError("EdaConfig counts and dpi must be positive")
        if self.prior_strength <= 0 or not 0.5 < self.length_clip_quantile <= 1:
            raise ValueError("prior_strength must be positive and length_clip_quantile in (0.5, 1]")


@dataclass(frozen=True)
class ClassicalConfig:
    """Step 5 settings: TF-IDF / bag-of-words features and the classical models with their tuning grids."""

    ngram_range: Tuple[int, int] = (1, 2)   # bigrams catch "not good" and "waste of money"
    min_df: int = 5                         # a term must appear in at least 5 training reviews
    max_features: int = 200_000
    sublinear_tf: bool = True               # 1 + log(tf): a word repeated ten times is not ten times as strong
    c_grid: Tuple[float, ...] = (0.3, 1.0, 3.0)         # Logistic Regression and Linear SVM
    alpha_grid: Tuple[float, ...] = (0.1, 0.5, 1.0)     # Naive Bayes smoothing
    max_iter: int = 1000
    inference_sample: int = 5000            # reviews used to time prediction (vectorising included)
    inference_repeats: int = 3
    top_terms: int = 25
    error_rows: int = 100                   # most confident mistakes saved for reading
    sample_rows: int = 40_000               # size of the quick smoke-test run
    model_name: str = "classical_best"
    comparison_name: str = "model_comparison"
    results_name: str = "classical_results"
    errors_name: str = "classical_errors"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        low, high = self.ngram_range
        if not 1 <= low <= high:
            raise ValueError("ngram_range must be (low, high) with 1 <= low <= high")
        if min(self.min_df, self.max_features, self.max_iter, self.inference_sample, self.inference_repeats,
               self.top_terms, self.error_rows, self.sample_rows) < 1:
            raise ValueError("counts in ClassicalConfig must be positive")
        if not self.c_grid or not self.alpha_grid or min(self.c_grid + self.alpha_grid) <= 0:
            raise ValueError("c_grid and alpha_grid must be non-empty and positive")


@dataclass(frozen=True)
class TransformerConfig:
    """Step 6 settings: fine-tuning DistilBERT with the Hugging Face Trainer."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 256                   # covers well over 95% of reviews (95th percentile is 215 words)
    epochs: int = 2
    train_batch_size: int = 32
    eval_batch_size: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06           # share of training steps spent warming the learning rate up
    logging_steps: int = 200
    bf16: bool = True                       # RTX 30/40/50-series GPUs support bfloat16; faster and half the memory
    sample_rows: int = 30_000               # size of the quick smoke-test run
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    inference_sample: int = 2000
    inference_repeats: int = 3
    inference_batch_size: int = 128
    error_rows: int = 100
    slice_fraction: float = 1 / 3           # share of the test set treated as "oldest" and as "newest"
    model_dir: str = "distilbert_sentiment"
    results_name: str = "transformer_results"
    comparison_name: str = "model_comparison_all"
    errors_name: str = "transformer_errors"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if min(self.max_length, self.epochs, self.train_batch_size, self.eval_batch_size, self.logging_steps,
               self.sample_rows, self.inference_sample, self.inference_repeats, self.inference_batch_size,
               self.error_rows) < 1:
            raise ValueError("counts in TransformerConfig must be positive")
        if self.learning_rate <= 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("learning_rate must be positive and warmup_fraction in [0, 1)")
        if not 0 < self.threshold_min < self.threshold_max < 1 or self.threshold_step <= 0:
            raise ValueError("thresholds need 0 < min < max < 1 and a positive step")
        if not 0 < self.slice_fraction <= 0.5:
            raise ValueError("slice_fraction must be in (0, 0.5]")


@dataclass(frozen=True)
class TopicConfig:
    """Step 7 settings: LDA (scikit-learn) and BERTopic, both scored with c_v coherence."""

    sample_rows: int = 100_000              # reviews modelled per run (seeded); embedding 390k reviews adds time, not insight
    sweep_rows: int = 40_000                # smaller sample used only to choose the number of LDA topics
    k_grid: Tuple[int, ...] = (6, 8, 10, 12, 15, 20)
    lda_passes: int = 5                     # passes over the data (scikit-learn max_iter)
    lda_iterations: int = 100               # inner iterations per mini-batch
    lda_batch_size: int = 2048
    workers: int = 3                        # parallel jobs for LDA
    no_below: int = 20                      # drop words found in fewer reviews than this
    no_above: float = 0.5                   # drop words found in more than half of the reviews
    keep_n: int = 20_000                    # vocabulary size cap
    coherence_window: int = 110
    top_words: int = 10
    coherence_rows: int = 20_000            # c_v is slow, so it is measured on a seeded subset of reviews
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_batch_size: int = 256
    min_topic_size: int = 100               # smallest BERTopic cluster
    umap_neighbors: int = 15
    umap_components: int = 5
    smoke_rows: int = 15_000                # quick-check run
    smoke_k_grid: Tuple[int, ...] = (6, 10)
    smoke_min_topic_size: int = 30
    summary_name: str = "topic_summary"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if min(self.sample_rows, self.sweep_rows, self.lda_passes, self.lda_iterations, self.lda_batch_size, self.workers, self.coherence_window,
               self.no_below, self.keep_n, self.top_words, self.coherence_rows, self.embed_batch_size,
               self.min_topic_size, self.umap_neighbors, self.umap_components, self.smoke_rows,
               self.smoke_min_topic_size) < 1:
            raise ValueError("counts in TopicConfig must be positive")
        if not self.k_grid or not self.smoke_k_grid or min(self.k_grid + self.smoke_k_grid) < 2:
            raise ValueError("k_grid values must be 2 or more")
        if not 0 < self.no_above <= 1:
            raise ValueError("no_above must be in (0, 1]")

    def for_smoke(self) -> "TopicConfig":
        """A copy sized for a few-minute check of the setup."""
        from dataclasses import replace
        return replace(self, sample_rows=self.smoke_rows, sweep_rows=min(self.sweep_rows, self.smoke_rows),
                       k_grid=self.smoke_k_grid, min_topic_size=self.smoke_min_topic_size,
                       coherence_rows=min(self.coherence_rows, self.smoke_rows), no_below=min(self.no_below, 5))


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
    """Whole-word keywords per aspect, with common inflections (matching is on raw sentences, not lemmas).

    ``listing`` and ``safety`` were added after Steps 4 and 7: "description", "picture" and "refund" kept appearing
    among the words most typical of negative reviews, and a China-made pet-treat topic was over 50% negative.
    """
    return {
        "taste": ("taste", "tastes", "tasted", "tasty", "tasteless", "flavor", "flavors", "flavorful", "delicious", "bland",
                  "sweet", "sweeter", "bitter", "salty", "aroma"),
        "price": ("price", "priced", "pricey", "cheap", "cheaper", "expensive", "cost", "costs", "value", "worth",
                  "overpriced", "deal", "bargain"),
        "shipping": ("shipping", "shipped", "ship", "delivery", "delivered", "arrived", "arrive", "late", "delay", "delayed"),
        "packaging": ("packaging", "packed", "damaged", "dented", "broken", "leaked", "leaking", "leak", "crushed", "sealed",
                      "seal", "torn", "tampered"),
        "freshness": ("fresh", "stale", "expired", "expiration", "expiry", "rancid", "moldy", "mold", "spoiled"),
        "quality": ("quality", "ingredients", "artificial", "organic", "natural", "texture"),
        "health": ("healthy", "unhealthy", "sugar", "calories", "allergy", "allergic", "gluten", "diet", "protein", "sodium"),
        "service": ("service", "refund", "refunded", "return", "returned", "seller", "support", "replacement", "replaced",
                    "customer"),
        "listing": ("description", "described", "picture", "pictures", "photo", "advertised", "listing", "listed", "label",
                    "labeled", "misleading"),
        "safety": ("sick", "vomit", "vomiting", "diarrhea", "recall", "fda", "illness", "poison", "toxic", "died"),
    }


@dataclass(frozen=True)
class AspectConfig:
    """Step 8 settings: aspect keywords, sentence scoring and drop detection."""

    keywords: Dict[str, Tuple[str, ...]] = field(default_factory=_default_aspects)
    min_sentence_words: int = 3
    neutral_low: float = 0.25               # sentences with P(positive) between low and high count as neutral / mixed
    neutral_high: float = 0.75
    max_length: int = 128                   # tokens per sentence for the transformer
    batch_size: int = 256
    sample_rows: int = 20_000               # size of the quick smoke-test run
    examples_per_aspect: int = 5
    min_period_sentences: int = 100         # a month with fewer polar sentences for an aspect is not trended
    min_period_reviews: int = 200           # a month with fewer reviews is not trended
    drop_window: int = 12                   # months in the trailing baseline
    drop_min_periods: int = 6
    drop_z: float = 2.0                     # flag a month this many standard deviations worse than its baseline
    summary_name: str = "insights_summary"

    def __post_init__(self) -> None:
        """Fail loudly on inconsistent settings."""
        if not self.keywords or any(not words for words in self.keywords.values()):
            raise ValueError("every aspect needs at least one keyword")
        if not 0 < self.neutral_low < self.neutral_high < 1:
            raise ValueError("need 0 < neutral_low < neutral_high < 1")
        if min(self.min_sentence_words, self.max_length, self.batch_size, self.sample_rows, self.examples_per_aspect,
               self.min_period_sentences, self.min_period_reviews, self.drop_window, self.drop_min_periods) < 1 \
                or self.drop_min_periods > self.drop_window or self.drop_z <= 0:
            raise ValueError("counts must be positive, drop_min_periods <= drop_window and drop_z > 0")


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
    eda: EdaConfig = field(default_factory=EdaConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)

"""Step 3: turn raw review text into model-ready tokens.

Two text versions are kept side by side: the untouched ``Text`` column for transformers, and ``text_clean``
(lower-case lemmas without stop words, negations kept) for TF-IDF, topic models and word clouds.
"""
from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from reviews.config import TextConfig

LOGGER = logging.getLogger(__name__)

_TAGS = re.compile(r"<[^>]+>")
_URLS = re.compile(r"(?:https?://|www\.)\S+")
_NON_LETTERS = re.compile(r"[^a-zA-Z\s]")
_NON_LETTERS_KEEP_DIGITS = re.compile(r"[^a-zA-Z0-9\s]")
_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = re.compile(r"[‘’´`]")

# Specific forms first, then the generic endings, so "won't" becomes "will not" and not "wo not".
_CONTRACTIONS = (
    (re.compile(r"\bcan't\b", re.I), "cannot"),
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"\bshan't\b", re.I), "shall not"),
    (re.compile(r"\bain't\b", re.I), "is not"),
    (re.compile(r"\blet's\b", re.I), "let us"),
    (re.compile(r"n't\b", re.I), " not"),
    (re.compile(r"'re\b", re.I), " are"),
    (re.compile(r"'ve\b", re.I), " have"),
    (re.compile(r"'ll\b", re.I), " will"),
    (re.compile(r"'m\b", re.I), " am"),
    (re.compile(r"'d\b", re.I), " would"),
    (re.compile(r"'s\b", re.I), ""),            # possessive or "is": both carry no sentiment
)

# Emoticons seen in 2007-2012 reviews. Plain words so they survive the letters-only filter.
_EMOTICONS = (
    (re.compile(r"(?::|=|;)-?\)+|:-?d\b|\^_\^", re.I), " happyface "),
    (re.compile(r"(?::|=)-?\(+|:'\(|:-?/(?!/)", re.I), " sadface "),
)


def strip_markup(text: str) -> str:
    """Unescape HTML entities and remove tags such as ``<br />``."""
    return _TAGS.sub(" ", html.unescape(text))


def expand_contractions(text: str) -> str:
    """"don't" -> "do not", "I'm" -> "I am"; curly apostrophes are normalised first."""
    text = _APOSTROPHES.sub("'", text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text: object, cfg: TextConfig = TextConfig()) -> str:
    """Normalise one review: markup, URLs, emoticons, contractions, case, symbols, digits, spacing.

    Anything that is not a string becomes an empty string.
    """
    if not isinstance(text, str):
        return ""
    if cfg.remove_html:
        text = strip_markup(text)
    if cfg.remove_urls:
        text = _URLS.sub(" ", text)
    if cfg.map_emoticons:
        for pattern, replacement in _EMOTICONS:
            text = pattern.sub(replacement, text)
    if cfg.expand_contractions:
        text = expand_contractions(text)
    if cfg.lowercase:
        text = text.lower()
    text = (_NON_LETTERS if cfg.remove_digits else _NON_LETTERS_KEEP_DIGITS).sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def stopword_set(cfg: TextConfig = TextConfig()) -> frozenset:
    """The usual English stop words minus the words that change sentiment (``cfg.keep_words``)."""
    return frozenset(ENGLISH_STOP_WORDS) - frozenset(cfg.keep_words)


def filter_tokens(tokens: Iterable[str], stop: frozenset, cfg: TextConfig = TextConfig()) -> List[str]:
    """Drop stop words and very short tokens."""
    return [t for t in tokens if len(t) >= cfg.min_token_length and not (cfg.remove_stopwords and t in stop)]


def load_spacy(cfg: TextConfig = TextConfig()) -> Any:
    """Load the spaCy pipeline with only what lemmatization needs (no parser, no entity recogniser).

    Raises:
        ImportError: If spaCy or the language model is not installed, with the install commands.
    """
    try:
        import spacy
        return spacy.load(cfg.spacy_model, disable=["parser", "ner"])
    except (ImportError, OSError) as error:
        raise ImportError("spaCy or its model is missing. Run: pip install spacy  and  "
                          f"python -m spacy download {cfg.spacy_model}") from error


def lemmatize(texts: Sequence[str], nlp: Any, cfg: TextConfig = TextConfig()) -> List[List[str]]:
    """Lemmas of each cleaned text, in order. ``nlp`` is a loaded spaCy pipeline (or anything with ``pipe``)."""
    n_process = cfg.n_process if len(texts) >= cfg.batch_size else 1  # worker start-up costs more than it saves
    docs = nlp.pipe(texts, batch_size=cfg.batch_size, n_process=n_process)
    return [[token.lemma_ for token in doc if not token.is_space] for doc in docs]


def preprocess(frame: pd.DataFrame, nlp: Any, cfg: TextConfig = TextConfig(), text_column: str = "Text") -> pd.DataFrame:
    """Add ``text_clean`` (lemmas joined by spaces) and ``n_tokens`` to a copy of ``frame``.

    The raw text column is left exactly as it was.

    Raises:
        ValueError: If ``text_column`` is missing.
    """
    if text_column not in frame.columns:
        raise ValueError(f"Column {text_column!r} not found")
    cleaned = frame[text_column].map(lambda t: clean_text(t, cfg))
    lemmas = lemmatize(cleaned.tolist(), nlp, cfg)
    stop = stopword_set(cfg)
    tokens = [filter_tokens(doc, stop, cfg) for doc in lemmas]
    return frame.assign(text_clean=[" ".join(t) for t in tokens],
                        n_tokens=pd.Series([len(t) for t in tokens], index=frame.index, dtype="int32"))


def text_report(frame: pd.DataFrame, cfg: TextConfig = TextConfig(), top: int = 20) -> Dict[str, Any]:
    """Checks that preprocessing kept what it should: negations, length, vocabulary and the commonest tokens."""
    tokens = frame["text_clean"].str.split()
    counts = tokens.explode().value_counts()
    keep = {w: int(counts.get(w, 0)) for w in ("not", "no", "never", "but", "very")}
    return {
        "rows": int(len(frame)),
        "empty_after_cleaning": int((frame["n_tokens"] == 0).sum()),
        "tokens_per_review": {"median": float(frame["n_tokens"].median()), "p95": float(frame["n_tokens"].quantile(0.95)),
                              "max": int(frame["n_tokens"].max())},
        "words_before_median": float(frame["n_words"].median()),
        "vocabulary_size": int(len(counts)),
        "tokens_seen_once": int((counts == 1).sum()),
        "kept_sentiment_words": keep,
        "top_tokens": {str(k): int(v) for k, v in counts.head(top).items()},
        "reviews_with_negation_share": round(float(tokens.map(lambda t: bool({"not", "no", "never"} & set(t))).mean()), 4),
    }

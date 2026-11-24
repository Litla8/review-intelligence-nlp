"""Step 7 topic modelling: LDA (scikit-learn) and BERTopic, compared with c_v coherence and topic diversity.

The pure helpers (sampling, tokenising, tables, scores) need neither BERTopic nor a GPU. BERTopic and its embedding
model are imported only when they run, with a message saying what to install if they are missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reviews.coherence import cv_coherence
from reviews.config import TopicConfig

LOGGER = logging.getLogger(__name__)
SEED = 42
SCOPES = ("all", "negative")


@dataclass
class TopicRun:
    """One fitted topic model: top words per topic, the topic of every review (-1 = none), and its quality scores."""

    name: str
    topic_words: Dict[int, List[str]]
    assignments: np.ndarray
    coherence: float
    extras: Dict[str, Any] = field(default_factory=dict)
    model: Any = None

    @property
    def n_topics(self) -> int:
        return len(self.topic_words)


def sample_reviews(frame: pd.DataFrame, scope: str, n: int) -> pd.DataFrame:
    """Reviews with usable cleaned text, restricted to ``scope`` and sampled with a fixed seed (order kept)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    missing = [c for c in ("text_clean", "Text", "sentiment", "Score") if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    keep = frame["text_clean"].fillna("").str.len().gt(0)
    if scope == "negative":
        keep &= frame["sentiment"].astype(str).eq("negative")
    chosen = frame.loc[keep]
    if chosen.empty:
        raise ValueError("No reviews left to model")
    return chosen.sample(n=min(n, len(chosen)), random_state=SEED).sort_index().reset_index(drop=True)


def tokenise(texts: pd.Series, drop: Iterable[str]) -> List[List[str]]:
    """Split cleaned text into tokens and remove the display stop words (negations, filler verbs) topic models do not need."""
    skip = frozenset(drop)
    return [[t for t in doc.split() if t not in skip] for doc in texts.fillna("").astype(str)]


def topic_diversity(topic_words: Sequence[Sequence[str]]) -> float:
    """Share of distinct words across all topics' top words (1.0 = no word is repeated between topics)."""
    flat = [w for words in topic_words for w in words]
    return round(len(set(flat)) / len(flat), 4) if flat else 0.0


def choose_k(scores: Mapping[int, float]) -> int:
    """The number of topics with the highest coherence; ties go to the smaller number (simpler model)."""
    if not scores:
        raise ValueError("no coherence scores to choose from")
    best = max(scores.values())
    return min(k for k, v in scores.items() if v == best)


def topic_table(frame: pd.DataFrame, assignments: np.ndarray, topic_words: Mapping[int, Sequence[str]],
                shown: int = 8) -> pd.DataFrame:
    """One row per topic: top words, reviews, share, mean stars and negative share, with ``lift`` over the overall negative share.

    A lift of 2 means the topic is twice as likely to appear in a negative review as an average review is to be negative.
    """
    if len(assignments) != len(frame):
        raise ValueError("assignments must have one entry per review")
    work = pd.DataFrame({"topic": assignments, "Score": frame["Score"].to_numpy(),
                         "negative": frame["sentiment"].astype(str).eq("negative").to_numpy()})
    work = work[work["topic"] >= 0]
    overall = float(frame["sentiment"].astype(str).eq("negative").mean())
    grouped = work.groupby("topic")
    table = pd.DataFrame({"reviews": grouped.size(), "mean_score": grouped["Score"].mean().round(2),
                          "negative_share": grouped["negative"].mean().round(4)})
    table["share"] = (table["reviews"] / len(frame)).round(4)
    table["lift"] = (table["negative_share"] / overall).round(2) if overall > 0 else np.nan
    table.insert(0, "top_words", [" ".join(topic_words.get(int(t), [])[:shown]) for t in table.index])
    table.index.name = "topic"
    return table.sort_values("negative_share", ascending=False).reset_index()


def _need(package: str, pip: str) -> ImportError:
    return ImportError(f"{package} is not installed. Run: pip install {pip}")


def build_matrix(tokens: Sequence[Sequence[str]], cfg: TopicConfig) -> Tuple[Any, Any]:
    """Word-count matrix (rare and ubiquitous words removed, vocabulary capped) and the fitted vectoriser."""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(token_pattern=r"\S+", lowercase=False, min_df=cfg.no_below, max_df=cfg.no_above,
                                 max_features=cfg.keep_n)
    try:
        matrix = vectorizer.fit_transform([" ".join(doc) for doc in tokens])
    except ValueError as exc:  # scikit-learn raises this when the filters remove every word
        raise ValueError("no vocabulary left after filtering; lower no_below or raise no_above") from exc
    return matrix, vectorizer


def train_lda(matrix: Any, k: int, cfg: TopicConfig) -> Any:
    """Fit online variational LDA with ``k`` topics (seeded)."""
    from sklearn.decomposition import LatentDirichletAllocation
    return LatentDirichletAllocation(n_components=k, learning_method="online", max_iter=cfg.lda_passes,
                                     max_doc_update_iter=cfg.lda_iterations, batch_size=cfg.lda_batch_size,
                                     n_jobs=cfg.workers, random_state=SEED).fit(matrix)


def lda_words(model: Any, vocabulary: Sequence[str], topn: int) -> Dict[int, List[str]]:
    """Top words of every LDA topic (largest weight first)."""
    top = np.argsort(-model.components_, axis=1, kind="stable")[:, :topn]
    return {i: [str(vocabulary[j]) for j in row] for i, row in enumerate(top)}


def lda_assign(model: Any, matrix: Any) -> np.ndarray:
    """Most likely topic of each review; a review with no vocabulary words left gets -1."""
    best = model.transform(matrix).argmax(axis=1)
    empty = np.asarray(matrix.sum(axis=1)).ravel() == 0
    return np.where(empty, -1, best).astype(np.int64)


def run_lda(tokens: Sequence[Sequence[str]], cfg: TopicConfig) -> TopicRun:
    """Choose the number of topics by c_v on a smaller sample, then fit the final LDA on all ``tokens``."""
    matrix, vectorizer = build_matrix(tokens, cfg)
    vocabulary = vectorizer.get_feature_names_out()
    sweep_n = min(cfg.sweep_rows, len(tokens))
    sweep: Dict[int, float] = {}
    for k in cfg.k_grid:
        model = train_lda(matrix[:sweep_n], k, cfg)
        words = list(lda_words(model, vocabulary, cfg.top_words).values())
        sweep[k] = round(cv_coherence(words, tokens[:cfg.coherence_rows], cfg.coherence_window), 4)
        LOGGER.info("LDA k=%d: c_v %.4f", k, sweep[k])
    best_k = choose_k(sweep)
    model = train_lda(matrix, best_k, cfg)
    words = lda_words(model, vocabulary, cfg.top_words)
    score = cv_coherence(list(words.values()), tokens[:cfg.coherence_rows], cfg.coherence_window)
    return TopicRun("lda", words, lda_assign(model, matrix), round(score, 4),
                    {"coherence_sweep": sweep, "chosen_k": best_k, "vocabulary": int(len(vocabulary))}, model)


def run_bertopic(frame: pd.DataFrame, tokens: Sequence[Sequence[str]], cfg: TopicConfig,
                 drop: Iterable[str]) -> TopicRun:
    """BERTopic: embed the raw reviews on the GPU, cluster with UMAP + HDBSCAN, and name topics from the cleaned words.

    Embeddings use the raw text (sentence-transformers handles case and punctuation); the topic words come from
    ``text_clean`` so they are lemmas that match the LDA vocabulary and the coherence texts.
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise _need("bertopic and sentence-transformers", "bertopic sentence-transformers") from exc
    encoder = SentenceTransformer(cfg.embedding_model)
    embeddings = encoder.encode(frame["Text"].astype(str).tolist(), batch_size=cfg.embed_batch_size,
                                show_progress_bar=True, normalize_embeddings=True)
    umap_model = UMAP(n_neighbors=cfg.umap_neighbors, n_components=cfg.umap_components, min_dist=0.0, metric="cosine",
                      random_state=SEED)
    vectorizer = CountVectorizer(token_pattern=r"\S+", lowercase=False, stop_words=list(drop), min_df=5)
    topic_model = BERTopic(embedding_model=None, umap_model=umap_model, vectorizer_model=vectorizer,
                           min_topic_size=cfg.min_topic_size, top_n_words=cfg.top_words, calculate_probabilities=False,
                           verbose=False)
    docs = frame["text_clean"].astype(str).tolist()
    assignments, _ = topic_model.fit_transform(docs, embeddings)
    assignments = np.asarray(assignments, dtype=np.int64)
    words = {int(t): [w for w, _ in topic_model.get_topic(t)][:cfg.top_words] for t in sorted(set(assignments)) if t != -1}
    score = cv_coherence(list(words.values()), tokens[:cfg.coherence_rows], cfg.coherence_window) if words else 0.0
    return TopicRun("bertopic", words, assignments, round(score, 4),
                    {"outlier_share": round(float((assignments == -1).mean()), 4)}, topic_model)

"""Step 4 analysis: review length, word and phrase frequencies, distinctive words and the rating trend.

Every function takes a DataFrame or Series and returns a DataFrame or Series, so it can be tested on a handful of
rows and reused in notebooks. Charts live in ``reviews/plots.py``.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from reviews.config import EdaConfig

LOGGER = logging.getLogger(__name__)
SEED = 42


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error if ``frame`` lacks any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def length_by_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Review length per sentiment: count, mean, median and 95th percentile of words and cleaned tokens."""
    require_columns(frame, ["sentiment", "n_words", "n_tokens"])
    grouped = frame.groupby("sentiment", observed=True)
    stats = grouped[["n_words", "n_tokens"]].agg(["mean", "median"])
    stats[("n_words", "p95")] = grouped["n_words"].quantile(0.95)
    stats[("n_tokens", "p95")] = grouped["n_tokens"].quantile(0.95)
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    stats.insert(0, "reviews", grouped.size())
    return stats.round(1)


def _count_matrix(texts: pd.Series, ngram: int, min_df: int) -> Optional[tuple]:
    """Count n-grams with scikit-learn. Returns ``(totals, vocabulary)`` or None when nothing is left to count."""
    vectorizer = CountVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(ngram, ngram), min_df=min_df,
                                 dtype=np.int64)
    try:
        matrix = vectorizer.fit_transform(texts.fillna("").astype(str))
    except ValueError:  # "empty vocabulary": every review was empty or every n-gram was too rare
        return None
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    return totals, vectorizer.get_feature_names_out()


def token_frequencies(texts: pd.Series, min_count: int = 1) -> pd.Series:
    """Total count of every cleaned token (term frequency, not document frequency), largest first."""
    counted = _count_matrix(texts, 1, 1)
    if counted is None:
        return pd.Series(dtype="int64", name="count")
    totals, vocabulary = counted
    series = pd.Series(totals, index=vocabulary, name="count").sort_values(ascending=False, kind="stable")
    return series[series >= min_count]


def top_words(texts: pd.Series, top_n: int, exclude: Iterable[str] = ()) -> pd.Series:
    """The ``top_n`` most frequent tokens after removing ``exclude`` (the display stop words)."""
    freqs = token_frequencies(texts)
    return freqs.drop(labels=list(set(exclude) & set(freqs.index))).head(top_n)


def top_ngrams(texts: pd.Series, n: int, top_n: int, exclude: Iterable[str] = (), min_df: int = 5,
               candidates: int = 2000) -> pd.Series:
    """Most frequent word pairs (``n=2``) or triples (``n=3``).

    A phrase is dropped only when *every* word in it is in ``exclude``, so "not good" and "very disappointed" stay
    while "just make" goes. Only the ``candidates`` most frequent phrases are checked, which keeps the loop tiny.
    """
    if n < 2:
        raise ValueError("n must be 2 or more; use top_words for single words")
    counted = _count_matrix(texts, n, min_df)
    if counted is None:
        return pd.Series(dtype="int64", name="count")
    totals, vocabulary = counted
    order = np.argsort(-totals, kind="stable")[:candidates]
    skip = set(exclude)
    keep = [i for i in order if not all(word in skip for word in vocabulary[i].split())]
    chosen = keep[:top_n]
    return pd.Series(totals[chosen], index=vocabulary[chosen], name="count")


def distinctive_words(counts_a: pd.Series, counts_b: pd.Series, min_count: int, prior_strength: float,
                      exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Rank words by how much more typical they are of group A than group B (log-odds z-score with a prior).

    This is the "informative Dirichlet prior" method of Monroe, Colaresi and Quinn (2008). A plain frequency chart
    shows "taste" and "flavor" for every group; the z-score shows what is special about each. Positive ``z`` means
    typical of A, negative of B. The prior is the overall word frequency, so rare words cannot win on luck.
    """
    words = counts_a.index.union(counts_b.index)
    a = counts_a.reindex(words, fill_value=0).to_numpy(dtype=float)
    b = counts_b.reindex(words, fill_value=0).to_numpy(dtype=float)
    total = a + b
    prior = prior_strength * total / total.sum()
    prior_total = prior.sum()
    a_total, b_total = a.sum(), b.sum()
    log_odds_a = np.log((a + prior) / (a_total + prior_total - a - prior))
    log_odds_b = np.log((b + prior) / (b_total + prior_total - b - prior))
    z = (log_odds_a - log_odds_b) / np.sqrt(1 / (a + prior) + 1 / (b + prior))
    result = pd.DataFrame({"count_a": a.astype("int64"), "count_b": b.astype("int64"), "z": z}, index=words)
    result = result[(result["count_a"] + result["count_b"] >= min_count)]
    result = result.drop(index=list(set(exclude) & set(result.index)))
    return result.sort_values("z", ascending=False, kind="stable")


def class_word_counts(frame: pd.DataFrame, text_column: str = "text_clean") -> Dict[str, pd.Series]:
    """Token counts for every sentiment present, computed once and reused for charts, clouds and z-scores."""
    require_columns(frame, ["sentiment", text_column])
    return {str(name): token_frequencies(group[text_column])
            for name, group in frame.groupby("sentiment", observed=True)}


def sample_per_class(frame: pd.DataFrame, max_docs: int) -> pd.DataFrame:
    """At most ``max_docs`` reviews per sentiment, chosen with a fixed seed so reruns give the same phrases."""
    require_columns(frame, ["sentiment"])
    parts = [group.sample(n=min(max_docs, len(group)), random_state=SEED)
             for _, group in frame.groupby("sentiment", observed=True)]
    return pd.concat(parts).sort_index()


def monthly_trend(frame: pd.DataFrame, min_reviews: int) -> pd.DataFrame:
    """Reviews, mean stars and the share of negative and positive reviews per calendar month.

    Months with fewer than ``min_reviews`` reviews are left out (their shares swing wildly on a few reviews).
    """
    require_columns(frame, ["review_date", "Score", "sentiment"])
    month = frame["review_date"].dt.to_period("M").dt.to_timestamp()
    shares = pd.crosstab(month, frame["sentiment"].astype(str), normalize="index")
    out = pd.DataFrame({
        "n_reviews": month.value_counts().sort_index(),
        "mean_score": frame.groupby(month)["Score"].mean(),
        "share_negative": shares.get("negative", 0.0),
        "share_positive": shares.get("positive", 0.0),
    })
    out.index.name = "month"
    return out[out["n_reviews"] >= min_reviews].round(4)


def yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Reviews and sentiment shares per year, for the written summary."""
    require_columns(frame, ["review_date", "sentiment"])
    year = frame["review_date"].dt.year.rename("year")
    shares = pd.crosstab(year, frame["sentiment"].astype(str), normalize="index").round(4)
    shares.insert(0, "n_reviews", year.value_counts().sort_index())
    return shares


def build_summary(frame: pd.DataFrame, cfg: EdaConfig, counts: Dict[str, pd.Series]) -> Dict[str, object]:
    """Everything the README and the Step 8 report need, as plain JSON-ready numbers."""
    stop = cfg.display_stopwords
    summary: Dict[str, object] = {
        "rows": int(len(frame)),
        "length": length_by_sentiment(frame).to_dict(orient="index"),
        "yearly": yearly_summary(frame).to_dict(orient="index"),
        "top_words": {name: top_words_from_counts(c, cfg.top_n, stop).to_dict() for name, c in counts.items()},
    }
    if {"negative", "positive"} <= counts.keys():
        table = distinctive_words(counts["negative"], counts["positive"], cfg.min_count_distinctive,
                                  cfg.prior_strength, stop)
        summary["distinctive_negative"] = table.head(cfg.top_n)["z"].round(2).to_dict()
        summary["distinctive_positive"] = table.tail(cfg.top_n).iloc[::-1]["z"].round(2).to_dict()
    return summary


def top_words_from_counts(counts: pd.Series, top_n: int, exclude: Iterable[str] = ()) -> pd.Series:
    """Same as :func:`top_words` but for counts that were already computed."""
    return counts.drop(labels=list(set(exclude) & set(counts.index))).head(top_n)

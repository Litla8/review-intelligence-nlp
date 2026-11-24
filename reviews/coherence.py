"""c_v topic coherence without Gensim.

Follows the c_v recipe of Röder, Both and Hinneburg (2015) as Gensim implements it: boolean sliding windows of 110
tokens (a review shorter than that is one window), NPMI between every pair of a topic's top words, then the cosine
similarity between each word's NPMI vector and the sum of all the topic's vectors, averaged over words and topics.
Gensim does not install on Python 3.14, so this is a re-implementation; values should be close to Gensim's but are not
guaranteed to be identical.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12  # the same smoothing constant Gensim uses so that log(0) never happens


def window_presence(docs: Sequence[Sequence[str]], vocabulary: Dict[str, int], window: int) -> np.ndarray:
    """Boolean matrix (windows x vocabulary words): does the word occur in the window?

    A document of at most ``window`` tokens is one window; a longer one gives one window per start position.
    """
    rows: List[np.ndarray] = []
    size = len(vocabulary)
    for doc in docs:
        positions = np.array([i for i, tok in enumerate(doc) if tok in vocabulary], dtype=np.int64)
        ids = np.array([vocabulary[doc[i]] for i in positions], dtype=np.int64)
        if len(doc) <= window:
            row = np.zeros(size, dtype=bool)
            row[ids] = True
            rows.append(row)
            continue
        for start in range(len(doc) - window + 1):
            lo, hi = np.searchsorted(positions, [start, start + window])
            row = np.zeros(size, dtype=bool)
            row[ids[lo:hi]] = True
            rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, size), dtype=bool)


def npmi_matrix(presence: np.ndarray) -> np.ndarray:
    """Normalised pointwise mutual information for every pair of vocabulary words (diagonal is 1)."""
    n_windows = presence.shape[0]
    if n_windows == 0:
        raise ValueError("no text windows to measure coherence on")
    counts = presence.T.astype(np.float64) @ presence.astype(np.float64)
    joint = counts / n_windows
    single = np.diag(joint)
    return np.log((joint + EPSILON) / np.outer(single, single).clip(min=EPSILON)) / -np.log(joint + EPSILON)


def cv_coherence(topics: Sequence[Sequence[str]], docs: Sequence[Sequence[str]], window: int = 110) -> float:
    """Mean c_v coherence of ``topics`` (lists of top words) measured on tokenised ``docs``.

    Words that never occur in ``docs`` are dropped; a topic with fewer than two words left is skipped. Returns 0.0
    when no topic can be scored.
    """
    if window < 1:
        raise ValueError("window must be positive")
    vocabulary: Dict[str, int] = {}
    for words in topics:
        for w in words:
            vocabulary.setdefault(w, len(vocabulary))
    if not vocabulary:
        return 0.0
    presence = window_presence(docs, vocabulary, window)
    seen = presence.any(axis=0)
    npmi = npmi_matrix(presence)
    scores = []
    for words in topics:
        idx = sorted({vocabulary[w] for w in words if seen[vocabulary[w]]})
        if len(idx) < 2:
            continue
        sub = npmi[np.ix_(idx, idx)]
        total = sub.sum(axis=0)                       # the vector for "all of the topic's words together"
        denom = np.linalg.norm(sub, axis=1) * np.linalg.norm(total)
        scores.append(float(np.mean((sub @ total) / np.where(denom == 0, 1.0, denom))))
    return float(np.mean(scores)) if scores else 0.0

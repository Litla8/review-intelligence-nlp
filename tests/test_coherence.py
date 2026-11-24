"""Tests for the Gensim-free c_v coherence."""
from __future__ import annotations

import numpy as np
import pytest

from reviews import coherence


def test_window_presence_short_docs_are_one_window_each():
    vocab = {"a": 0, "b": 1}
    out = coherence.window_presence([["a", "x", "b"], ["b"], ["x"]], vocab, window=110)
    assert out.tolist() == [[True, True], [False, True], [False, False]]


def test_window_presence_slides_over_long_docs():
    vocab = {"a": 0, "b": 1}
    doc = ["a", "x", "x", "x", "b"]  # window 3 -> starts 0, 1, 2
    out = coherence.window_presence([doc], vocab, window=3)
    assert out.tolist() == [[True, False], [False, False], [False, True]]


def test_npmi_is_one_for_words_that_always_appear_together():
    presence = np.array([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=bool)
    assert np.allclose(coherence.npmi_matrix(presence), 1.0)


def test_npmi_is_strongly_negative_for_words_that_never_meet():
    presence = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    npmi = coherence.npmi_matrix(presence)
    assert npmi[0, 0] == pytest.approx(1.0) and npmi[0, 1] < -0.9


def test_cv_is_one_for_perfectly_cooccurring_topic():
    docs = [["a", "b", "c"]] * 5 + [["x", "y"]] * 5
    assert coherence.cv_coherence([["a", "b", "c"]], docs) == pytest.approx(1.0)


def test_coherent_topics_beat_mixed_topics():
    docs = [["a", "b", "c"]] * 10 + [["x", "y", "z"]] * 10
    good = coherence.cv_coherence([["a", "b", "c"], ["x", "y", "z"]], docs)
    mixed = coherence.cv_coherence([["a", "x", "c"], ["b", "y", "z"]], docs)
    assert good > mixed and good > 0.9 and mixed < 0.5


def test_unseen_words_and_tiny_topics_are_skipped():
    docs = [["a", "b"]] * 4
    assert coherence.cv_coherence([["a", "b", "ghost"]], docs) == pytest.approx(1.0)  # ghost is dropped
    assert coherence.cv_coherence([["a", "ghost"]], docs) == 0.0                         # only one usable word
    assert coherence.cv_coherence([], docs) == 0.0


def test_validation():
    with pytest.raises(ValueError, match="window"):
        coherence.cv_coherence([["a", "b"]], [["a"]], window=0)
    with pytest.raises(ValueError, match="no text windows"):
        coherence.cv_coherence([["a", "b"]], [])

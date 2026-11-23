"""Tests for reviews.labels."""
from __future__ import annotations

import pandas as pd
import pytest

from reviews.labels import class_balance, stars_to_sentiment, to_binary


def test_star_mapping_matches_the_brief() -> None:
    result = stars_to_sentiment(pd.Series([1, 2, 3, 4, 5]))
    assert result.astype(str).tolist() == ["negative", "negative", "neutral", "positive", "positive"]


def test_float_and_string_ratings_are_accepted_and_index_is_kept() -> None:
    result = stars_to_sentiment(pd.Series(["5", 1.0], index=[10, 20]))
    assert result.index.tolist() == [10, 20] and result.astype(str).tolist() == ["positive", "negative"]


@pytest.mark.parametrize("bad", [[0, 3], [6], [None, 3], ["x"]])
def test_invalid_ratings_raise(bad) -> None:
    with pytest.raises(ValueError, match="1 to 5"):
        stars_to_sentiment(pd.Series(bad))


def test_binary_drops_neutral() -> None:
    sentiment = stars_to_sentiment(pd.Series([1, 3, 5, 3]))
    assert to_binary(sentiment).tolist() == ["negative", "positive"]


def test_class_balance_includes_empty_classes_and_ratio() -> None:
    sentiment = stars_to_sentiment(pd.Series([5, 5, 5, 1]))
    out = class_balance(sentiment)
    assert out["counts"] == {"negative": 1, "neutral": 0, "positive": 3}
    assert out["imbalance_ratio"]["majority_to_minority"] == 3.0
    with pytest.raises(ValueError):
        class_balance(sentiment.iloc[0:0])

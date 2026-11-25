"""Tests for reviews.analyzer (scoring functions are fakes)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reviews import analyzer as an
from reviews.config import AppConfig, AspectConfig


def fake(texts):
    bad = ("stale", "late", "broken", "awful", "terrible", "damaged")
    return np.array([0.05 if any(b in t.lower() for b in bad) else 0.95 for t in texts])


SC = an.Scorers(fake, fake, 0.5)


def test_guess_column_and_dates() -> None:
    assert an.guess_column(["Id", "Review", "Stars"], an.TEXT_GUESSES) == "Review"
    assert an.guess_column(["a"], an.TEXT_GUESSES) is None
    assert an.parse_dates(pd.Series([1300000000, 1310000000])).dt.year.tolist() == [2011, 2011]
    assert an.parse_dates(pd.Series(["2020-01-05", "bad"])).isna().tolist() == [False, True]


def test_analyze_review() -> None:
    out = an.analyze_review("The coffee was stale and awful. Shipping was fast and the price was great.", SC)
    assert out["label"] == "negative" or out["label"] in ("positive", "mixed")
    sentiments = {r["sentence"][:10]: r["sentiment"] for r in out["aspect_sentences"]}
    assert "negative" in sentiments.values() and "positive" in sentiments.values()
    with pytest.raises(ValueError):
        an.analyze_review("   ", SC)


def _frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    stars = rng.integers(1, 6, n)
    text = [("The coffee was stale and the shipping was late today." if s <= 2
             else "Great taste and a fair price for this coffee.") for s in stars]
    return pd.DataFrame({"Review": text, "Stars": stars, "When": pd.date_range("2018-01-01", periods=n, freq="D")})


def test_analyze_frame_with_ratings_and_dates() -> None:
    app = AppConfig(max_rows=300)
    res = an.analyze_frame(_frame(), "Review", SC, app, replace(AspectConfig(), min_period_reviews=5),
                           rating_column="Stars", date_column="When")
    assert res.metrics["sampled"] and res.metrics["reviews_analysed"] == 300
    assert res.metrics["model_vs_stars_macro_f1"] > 0.9
    assert {"shipping", "taste"} <= set(res.aspects["aspect"])
    assert res.monthly is not None and res.flags is not None
    assert (res.examples["aspect"] == "shipping").any()


def test_analyze_frame_without_ratings_or_dates() -> None:
    res = an.analyze_frame(_frame(50), "Review", SC)
    assert res.monthly is None and "model_vs_stars_macro_f1" not in res.metrics
    assert res.metrics["negative_share"] + res.metrics["positive_share"] == pytest.approx(1.0)


def test_analyze_frame_errors() -> None:
    with pytest.raises(ValueError, match="not found"):
        an.analyze_frame(_frame(5), "Nope", SC)
    with pytest.raises(ValueError, match="No reviews"):
        an.analyze_frame(pd.DataFrame({"Review": ["", " "]}), "Review", SC)

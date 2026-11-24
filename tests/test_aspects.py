"""Step 8 tests with a fake sentence scorer (no GPU): lexicon, sentences, shares, trends, flags and the pipeline."""
from __future__ import annotations

import json
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from reviews import aspects as asp
from reviews import insights, plots
from reviews.config import AspectConfig, PathsConfig, ProjectConfig

CFG = AspectConfig()


def reviews_frame() -> pd.DataFrame:
    """Four reviews: two negative (shipping / taste complaints), two positive (taste / price praise)."""
    return pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "Text": ["The package arrived late and damaged. I want a refund! Great seller though.",
                 "Bland taste. <br />Terrible flavor and too sweet.",
                 "Delicious chocolate! The price is a great value for the quality.",
                 "Love it. Tastes wonderful and shipping was fast."],
        "Score": [1, 2, 5, 5], "sentiment": pd.Categorical(["negative", "negative", "positive", "positive"],
                                                          categories=["negative", "neutral", "positive"]),
        "review_date": pd.to_datetime(["2011-01-15", "2011-02-15", "2012-01-15", "2012-02-15"])})


def keyword_scorer(sentences):
    """Fake model: a sentence is negative when it contains a clearly bad word, positive when it has a good one, else neutral."""
    bad = ("late", "damaged", "bland", "terrible", "refund")
    good = ("great", "delicious", "wonderful", "fast", "value")
    def p(sentence: str) -> float:
        words = set(sentence.lower().replace("!", " ").replace(".", " ").split())  # whole words: "chocolate" is not "late"
        return 0.05 if words & set(bad) else 0.95 if words & set(good) else 0.5
    return np.array([p(s) for s in sentences])


def test_lexicon_matches_whole_words_only():
    lex = asp.compile_lexicon(CFG)
    assert lex["shipping"].search("It arrived late.") and not lex["shipping"].search("I love chocolate")
    assert lex["taste"].search("Great TASTE") and not lex["taste"].search("tastelessness is rare")


def test_split_sentences_strips_html_drops_short_and_keeps_ids():
    out = asp.split_sentences(reviews_frame(), min_words=3)
    assert "<br" not in " ".join(out["sentence"]) and "Great seller though." in set(out["sentence"])
    assert "Bland taste." not in set(out["sentence"])  # two words: below the minimum
    assert (out["sentence"].str.split().str.len() >= 3).all() and set(out["Id"]) == {1, 2, 3, 4}
    assert out.loc[out["Id"] == 2, "review_sentiment"].eq("negative").all()
    with pytest.raises(ValueError, match="Missing columns"):
        asp.split_sentences(pd.DataFrame({"Text": ["x"]}))


def test_tag_aspects_flags_and_drops_untagged_sentences():
    sentences = asp.split_sentences(reviews_frame(), 2)
    tagged = asp.tag_aspects(sentences, asp.compile_lexicon(CFG))
    assert tagged["sentence"].map(lambda s: "Love it." in s).sum() == 0  # no aspect word in "Love it."
    row = tagged[tagged["sentence"].str.contains("arrived late")].iloc[0]
    assert row["shipping"] and not row["taste"]
    assert tagged[list(CFG.keywords)].any(axis=1).all()


def test_polarity_bands():
    assert list(asp.polarity(np.array([0.1, 0.25, 0.5, 0.75, 0.9]), CFG)) == [-1, -1, 0, 1, 1]


def test_score_sentences_validates_scorer_output():
    tagged = asp.tag_aspects(asp.split_sentences(reviews_frame(), 2), asp.compile_lexicon(CFG))
    scored = asp.score_sentences(tagged, keyword_scorer, CFG)
    assert {"p_positive", "polarity"} <= set(scored.columns) and len(scored) == len(tagged)
    with pytest.raises(ValueError, match="one probability"):
        asp.score_sentences(tagged, lambda s: np.zeros(1), CFG)
    assert asp.score_sentences(tagged.iloc[0:0], keyword_scorer, CFG).empty


def _scored() -> pd.DataFrame:
    sentences = asp.split_sentences(reviews_frame(), 2)
    return asp.score_sentences(asp.tag_aspects(sentences, asp.compile_lexicon(CFG)), keyword_scorer, CFG)


def test_aspect_summary_numbers():
    reviews = reviews_frame()
    summary = asp.aspect_summary(_scored(), reviews, list(CFG.keywords)).set_index("aspect")
    assert summary.loc["shipping", "reviews"] == 2 and summary.loc["shipping", "mention_rate"] == 0.5
    assert summary.loc["shipping", "neg_review_rate"] == 0.5 and summary.loc["shipping", "pos_review_rate"] == 0.5
    assert summary.loc["taste", "neg_review_rate"] == 0.5 and summary.loc["taste", "pos_review_rate"] == 1.0
    assert summary.loc["service", "neg_review_rate"] == 0.5 and summary.loc["service", "pos_review_rate"] == 0.0
    assert np.isnan(summary.loc["service", "pain_ratio"])  # no positive reviews mention it: ratio undefined
    assert summary.loc["packaging", "neg_sentence_share"] == 1.0 and summary.loc["packaging", "net_sentiment"] == -1.0


def test_aspect_summary_is_sorted_by_negative_reach():
    out = asp.aspect_summary(_scored(), reviews_frame(), list(CFG.keywords))
    assert out["neg_review_rate"].is_monotonic_decreasing


def test_aspect_trend_blanks_thin_periods():
    scored = _scored()
    thin = asp.aspect_trend(scored, ["taste"], "Y", min_sentences=100)
    assert thin["taste"].isna().all()
    yearly = asp.aspect_trend(scored, ["taste"], "Y", min_sentences=1)
    assert yearly.loc[pd.Timestamp("2011-01-01"), "taste"] == 1.0 and yearly.loc[pd.Timestamp("2012-01-01"), "taste"] == 0.0


def test_flag_drops_finds_injected_fall_and_rise():
    idx = pd.date_range("2010-01-01", periods=30, freq="MS")
    rng = np.random.default_rng(42)
    values = pd.Series(0.80 + rng.normal(0, 0.005, 30), index=idx)
    values.iloc[24] -= 0.10
    down = asp.flag_drops(values, 12, 6, 2.0, "down")
    assert list(down.index[down["flagged"]]) == [idx[24]]
    up = asp.flag_drops(values * -1 + 1.6, 12, 6, 2.0, "up")
    assert list(up.index[up["flagged"]]) == [idx[24]]
    assert not down["flagged"].iloc[:6].any()  # not enough history yet
    with pytest.raises(ValueError, match="direction"):
        asp.flag_drops(values, 12, 6, 2.0, "sideways")


def test_flag_drops_ignores_flat_series_and_uses_only_past():
    flat = pd.Series(0.8, index=pd.date_range("2010-01-01", periods=20, freq="MS"))
    assert not asp.flag_drops(flat, 12, 6, 2.0)["flagged"].any()  # zero spread: nothing to compare with


def test_monthly_overall_and_examples():
    overall = asp.monthly_overall(reviews_frame(), min_reviews=1)
    assert len(overall) == 4 and overall["share_positive"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert asp.monthly_overall(reviews_frame(), min_reviews=2).empty
    examples = asp.top_examples(_scored(), ["taste", "shipping"], 1)
    assert set(examples["aspect"]) <= {"taste", "shipping"} and (examples["p_positive"] <= 0.25).all()


def test_config_validation():
    for bad in ({"keywords": {}}, {"keywords": {"a": ()}}, {"neutral_low": 0.8}, {"drop_min_periods": 20}, {"drop_z": 0.0}):
        with pytest.raises(ValueError):
            AspectConfig(**bad)


def test_plots_render(tmp_path):
    result = insights.analyse(reviews_frame(), _cfg(tmp_path), keyword_scorer)
    figs = [plots.plot_aspect_pain_points(result["summary"]), plots.plot_aspect_trend(result["yearly"]),
            plots.plot_monthly_flags(result["flags"]["share_positive"], "t", "y")]
    for i, fig in enumerate(figs):
        assert plots.save_figure(fig, tmp_path / f"{i}.png", 50).stat().st_size > 0


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(paths=PathsConfig(root=tmp_path), aspects=AspectConfig(min_period_reviews=1, min_period_sentences=1,
                                                                                drop_min_periods=2, drop_window=3))


def test_run_requires_step3_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        insights.run(_cfg(tmp_path))


def test_run_end_to_end_with_fake_scorer(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()
    with mock.patch("reviews.insights.pd.read_parquet", return_value=reviews_frame()):
        summary = insights.run(cfg, sample=True, scorer=keyword_scorer)
    assert summary["reviews"] == 4 and {a["aspect"] for a in summary["aspects"]} == set(CFG.keywords)
    reports = cfg.paths.reports_path
    for name in ("aspect_summary_sample.csv", "aspect_trend_yearly_sample.csv", "monthly_trend_sample.csv",
                 "aspect_examples_sample.csv", "insights_summary_sample.json"):
        assert (reports / name).is_file()
    assert len(list(cfg.paths.figures_path.glob("*_sample.png"))) == 3
    assert "drop_flags" in json.loads((reports / "insights_summary_sample.json").read_text())


def test_scorer_loader_explains_missing_model_and_torch(tmp_path):
    cfg = ProjectConfig(paths=PathsConfig(root=tmp_path))
    with pytest.raises(FileNotFoundError, match="train_transformer"):
        insights.load_sentence_scorer(cfg)
    (cfg.paths.models_path / cfg.transformer.model_dir).mkdir(parents=True)
    with mock.patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="torch"):
            insights.load_sentence_scorer(cfg)

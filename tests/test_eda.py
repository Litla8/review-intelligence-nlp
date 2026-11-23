"""Step 4 tests: analysis functions, charts and the pipeline, on a small hand-built frame."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reviews import eda, explore, plots
from reviews.config import EdaConfig, ProjectConfig

NEG = ["not good bad taste", "bad taste not fresh", "not fresh bad smell", "stale not good"]
POS = ["great taste love", "love great flavor", "great coffee love", "good flavor great"]
NEU = ["okay taste but price high", "okay flavor but price"]


def make_frame(months: int = 4, per_month: int = 6) -> pd.DataFrame:
    """Deterministic frame: ``per_month`` reviews per month, cycling negative/positive/neutral texts."""
    rows = []
    for m in range(months):
        for i in range(per_month):
            kind = (m + i) % 3
            text = [NEG, POS, NEU][kind][i % len([NEG, POS, NEU][kind])]
            rows.append({"sentiment": ["negative", "positive", "neutral"][kind], "Score": [1, 5, 3][kind],
                         "text_clean": text, "n_words": len(text.split()) + 3, "n_tokens": len(text.split()),
                         "review_date": pd.Timestamp(2012, 1 + m, 10 + i)})
    frame = pd.DataFrame(rows)
    frame["sentiment"] = pd.Categorical(frame["sentiment"], categories=["negative", "neutral", "positive"])
    return frame


class FakeCloud:
    """Stands in for wordcloud.WordCloud: records the frequencies and returns a small image."""

    def __init__(self) -> None:
        self.calls = []

    def generate_from_frequencies(self, freqs):
        self.calls.append(freqs)
        return self

    def to_array(self):
        return np.zeros((4, 6, 3), dtype=np.uint8)


def test_length_by_sentiment_values():
    frame = pd.DataFrame({"sentiment": pd.Categorical(["negative", "negative", "positive"]),
                          "n_words": [10, 20, 30], "n_tokens": [5, 15, 25]})
    out = eda.length_by_sentiment(frame)
    assert out.loc["negative", "reviews"] == 2
    assert out.loc["negative", "n_words_median"] == 15.0
    assert out.loc["positive", "n_tokens_mean"] == 25.0


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing columns"):
        eda.length_by_sentiment(pd.DataFrame({"sentiment": ["x"]}))


def test_token_frequencies_counts_and_order():
    freqs = eda.token_frequencies(pd.Series(["not good", "not bad", "good"]))
    assert freqs.to_dict() == {"not": 2, "good": 2, "bad": 1}
    assert list(freqs.index[:2]) == ["good", "not"] or list(freqs.index[:2]) == ["not", "good"]
    assert eda.token_frequencies(pd.Series(["not good", "good"]), min_count=2).to_dict() == {"good": 2}


def test_token_frequencies_empty_input():
    assert eda.token_frequencies(pd.Series(["", None])).empty


def test_top_words_excludes_display_stopwords():
    out = eda.top_words(pd.Series(["not good good", "not bad"]), 5, exclude=["not"])
    assert "not" not in out.index and out["good"] == 2


def test_top_ngrams_keeps_phrases_with_a_content_word():
    texts = pd.Series(["not good taste"] * 3 + ["just make"] * 3)
    out = eda.top_ngrams(texts, 2, 5, exclude=["not", "just", "make"], min_df=1)
    assert "not good" in out.index and "just make" not in out.index
    assert out["not good"] == 3


def test_top_ngrams_rejects_unigrams_and_handles_empty():
    with pytest.raises(ValueError):
        eda.top_ngrams(pd.Series(["a b"]), 1, 5)
    assert eda.top_ngrams(pd.Series([""]), 2, 5, min_df=1).empty


def test_distinctive_words_sign_and_filters():
    a = pd.Series({"bad": 100, "taste": 100, "rare": 1})
    b = pd.Series({"great": 100, "taste": 100})
    out = eda.distinctive_words(a, b, min_count=5, prior_strength=50.0, exclude=["taste"])
    assert out.index[0] == "bad" and out.index[-1] == "great"
    assert out.loc["bad", "z"] > 0 > out.loc["great", "z"]
    assert "rare" not in out.index and "taste" not in out.index


def test_distinctive_words_shared_word_scores_near_zero():
    out = eda.distinctive_words(pd.Series({"taste": 100, "bad": 50}), pd.Series({"taste": 100, "good": 50}), 1, 10.0)
    assert abs(out.loc["taste", "z"]) < 0.5


def test_sample_per_class_caps_and_is_reproducible():
    frame = make_frame()
    one = eda.sample_per_class(frame, 3)
    two = eda.sample_per_class(frame, 3)
    assert one.groupby("sentiment", observed=True).size().max() == 3
    pd.testing.assert_frame_equal(one, two)


def test_monthly_trend_shares_and_min_filter():
    frame = make_frame(months=3, per_month=6)
    out = eda.monthly_trend(frame, min_reviews=1)
    assert len(out) == 3 and (out["n_reviews"] == 6).all()
    assert np.allclose(out["share_positive"], 1 / 3, atol=1e-3)  # the cycle gives each class 2 of 6 per month
    assert np.allclose(out["mean_score"], 3.0)  # (1 + 5 + 3) / 3
    assert eda.monthly_trend(frame, min_reviews=7).empty


def test_yearly_summary_rows_sum_to_one():
    out = eda.yearly_summary(make_frame())
    assert out.loc[2012, "n_reviews"] == 24
    assert out.drop(columns="n_reviews").sum(axis=1).round(3).eq(1.0).all()


def test_build_summary_has_expected_keys_and_json_safe():
    frame = make_frame()
    cfg = EdaConfig(min_count_distinctive=1, prior_strength=10.0)
    counts = eda.class_word_counts(frame)
    summary = eda.build_summary(frame, cfg, counts)
    json.dumps(summary)  # must not raise
    assert summary["rows"] == len(frame)
    assert "bad" in summary["distinctive_negative"] and "love" in summary["distinctive_positive"]
    assert "not" not in summary["top_words"]["negative"]  # display stop words are removed


def test_eda_config_validation():
    with pytest.raises(ValueError):
        EdaConfig(top_n=0)
    with pytest.raises(ValueError):
        EdaConfig(prior_strength=0)
    with pytest.raises(ValueError):
        EdaConfig(length_clip_quantile=0.2)


def test_charts_return_figures_and_save(tmp_path):
    frame = make_frame()
    counts = eda.class_word_counts(frame)
    top = {k: v.head(3) for k, v in counts.items()}
    figs = [plots.plot_length_distribution(frame), plots.plot_top_words(top), plots.plot_ngrams(top, "Bigrams"),
            plots.plot_distinctive(pd.Series({"bad": 3.0, "stale": 2.0}), pd.Series({"love": -3.0})),
            plots.plot_rating_trend(eda.monthly_trend(frame, 1)),
            plots.plot_wordclouds(counts, 10, cloud=FakeCloud())]
    path = plots.save_figure(figs[0], tmp_path / "sub" / "chart.png", dpi=50)
    assert path.is_file() and path.stat().st_size > 0
    assert all(hasattr(f, "savefig") for f in figs)


def test_wordcloud_uses_each_class_counts():
    cloud = FakeCloud()
    plots.plot_wordclouds({"negative": pd.Series({"bad": 3}), "positive": pd.Series({"love": 2})}, 10, cloud=cloud)
    assert cloud.calls == [{"bad": 3}, {"love": 2}]


def test_wordcloud_missing_package_gives_install_hint(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "wordcloud":
            raise ImportError("no wordcloud")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="pip install wordcloud"):
        plots.plot_wordclouds({"negative": pd.Series({"bad": 3})}, 10)


def test_analyse_writes_every_chart(tmp_path):
    cfg = ProjectConfig(eda=EdaConfig(min_count_distinctive=1, prior_strength=10.0, min_month_reviews=1, dpi=50))
    summary = explore.analyse(make_frame(), cfg, tmp_path, cloud=FakeCloud())
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"length_distribution.png", "top_words_by_sentiment.png", "wordclouds.png", "top_bigrams.png",
                     "top_trigrams.png", "distinctive_words.png", "sentiment_trend.png"}
    assert summary["months_plotted"] == 4


def test_run_requires_step3_output(tmp_path):
    cfg = ProjectConfig(paths=ProjectConfig().paths.__class__(root=tmp_path))
    with pytest.raises(FileNotFoundError, match="preprocess"):
        explore.run(cfg)


def test_run_end_to_end_idempotent(tmp_path):
    pytest.importorskip("pyarrow")
    base = ProjectConfig()
    cfg = ProjectConfig(paths=base.paths.__class__(root=tmp_path),
                        eda=EdaConfig(min_count_distinctive=1, prior_strength=10.0, min_month_reviews=1, dpi=50))
    cfg.paths.processed_path.mkdir(parents=True)
    make_frame().to_parquet(cfg.paths.text_path(False), index=False)
    first = explore.run(cfg, cloud=FakeCloud())
    summary_file = cfg.paths.eda_summary_path
    assert json.loads(summary_file.read_text()) == json.loads(json.dumps(first))
    stamp = (cfg.paths.figures_path / "sentiment_trend.png").stat().st_size
    second = explore.run(cfg, cloud=FakeCloud())
    assert second == first  # same inputs, same outputs
    assert (cfg.paths.figures_path / "sentiment_trend.png").stat().st_size == stamp

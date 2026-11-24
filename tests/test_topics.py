"""Step 7 tests. Gensim and BERTopic are replaced by fakes, so only the pipeline logic is exercised."""
from __future__ import annotations

import json
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from reviews import plots, run_topics, topics
from reviews.config import PathsConfig, ProjectConfig, TopicConfig


def make_frame(n: int = 120) -> pd.DataFrame:
    """Three themes cycling: shipping (mostly negative), taste (mostly positive), price (mixed)."""
    theme = np.arange(n) % 3
    words = np.array(["late damaged box refund", "great taste love flavor", "price cheap value money"])[theme]
    sentiment = np.where((theme == 0) & (np.arange(n) % 2 == 0), "negative", np.where(theme == 1, "positive", "positive"))
    return pd.DataFrame({"Id": np.arange(n), "Text": [f"Raw text {i}" for i in range(n)], "text_clean": words,
                         "sentiment": sentiment, "Score": np.where(sentiment == "negative", 1, 5)})


def test_sample_reviews_scope_seed_and_validation():
    frame = make_frame()
    frame.loc[0, "text_clean"] = ""
    a, b = topics.sample_reviews(frame, "all", 50), topics.sample_reviews(frame, "all", 50)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50 and (a["text_clean"].str.len() > 0).all()
    neg = topics.sample_reviews(frame, "negative", 1000)
    assert set(neg["sentiment"]) == {"negative"} and len(neg) == 19  # 20 negatives, one of which has empty text
    with pytest.raises(ValueError, match="scope"):
        topics.sample_reviews(frame, "positive", 10)
    with pytest.raises(ValueError, match="Missing columns"):
        topics.sample_reviews(frame.drop(columns="Score"), "all", 10)
    with pytest.raises(ValueError, match="No reviews"):
        topics.sample_reviews(frame[frame["sentiment"] == "positive"], "negative", 10)


def test_tokenise_removes_display_words_and_handles_missing():
    out = topics.tokenise(pd.Series(["not good coffee make", None, ""]), ["not", "make"])
    assert out == [["good", "coffee"], [], []]


def test_topic_diversity():
    assert topics.topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topics.topic_diversity([["a", "b"], ["a", "b"]]) == 0.5
    assert topics.topic_diversity([]) == 0.0


def test_choose_k_picks_best_and_smallest_on_ties():
    assert topics.choose_k({6: 0.4, 8: 0.55, 10: 0.5}) == 8
    assert topics.choose_k({6: 0.5, 8: 0.5}) == 6
    with pytest.raises(ValueError):
        topics.choose_k({})


def test_topic_table_values_and_order():
    frame = make_frame(12)
    assignments = np.arange(12) % 3  # same as theme
    table = topics.topic_table(frame, assignments, {0: ["late", "box"], 1: ["taste"], 2: ["price"]})
    assert list(table["topic"])[0] == 0  # shipping topic is the most negative
    row = table.set_index("topic")
    assert row.loc[0, "reviews"] == 4 and row.loc[0, "negative_share"] == 0.5 and row.loc[0, "top_words"] == "late box"
    overall = frame["sentiment"].eq("negative").mean()
    assert row.loc[0, "lift"] == round(0.5 / overall, 2) and row.loc[1, "negative_share"] == 0.0
    assert row["share"].sum() == pytest.approx(1.0, abs=1e-3)  # shares are rounded to 4 decimals


def test_topic_table_ignores_unassigned_and_checks_length():
    frame = make_frame(6)
    table = topics.topic_table(frame, np.array([0, 0, -1, -1, 1, 1]), {0: ["a"], 1: ["b"]})
    assert table["reviews"].sum() == 4 and set(table["topic"]) == {0, 1}
    with pytest.raises(ValueError, match="one entry per review"):
        topics.topic_table(frame, np.array([0]), {})


THEMES = [["late", "damaged", "box", "refund", "shipping"], ["great", "taste", "love", "flavor", "sweet"],
          ["price", "cheap", "value", "money", "worth"]]


def themed_tokens(n_per_theme: int = 60):
    """Documents that each draw only from one theme's words, so LDA has an obvious structure to find."""
    rng = np.random.default_rng(42)
    return [list(rng.choice(words, 6)) for _ in range(n_per_theme) for words in THEMES]


def test_build_matrix_filters_vocabulary_and_validates():
    cfg = TopicConfig(no_below=2, no_above=1.0)
    matrix, vectorizer = topics.build_matrix([["a", "b"], ["a", "c"], ["a", "b"]], cfg)
    assert set(vectorizer.get_feature_names_out()) == {"a", "b"} and matrix.shape == (3, 2)
    with pytest.raises(ValueError, match="no vocabulary"):
        topics.build_matrix([["a"], ["b"]], TopicConfig(no_below=5))


def test_lda_finds_the_three_themes_and_is_reproducible():
    tokens = themed_tokens()
    cfg = TopicConfig(no_below=2, workers=1, lda_passes=10)
    matrix, vectorizer = topics.build_matrix(tokens, cfg)
    model = topics.train_lda(matrix, 3, cfg)
    words = topics.lda_words(model, vectorizer.get_feature_names_out(), 5)
    found = {frozenset(w) for w in words.values()}
    assert found == {frozenset(t) for t in THEMES}
    assignments = topics.lda_assign(model, matrix)
    assert len(set(assignments)) == 3 and assignments[0] == assignments[3]  # same theme, same topic
    again = topics.lda_assign(topics.train_lda(matrix, 3, cfg), matrix)
    assert list(again) == list(assignments)


def test_lda_assign_marks_empty_documents():
    cfg = TopicConfig(no_below=2, workers=1)
    matrix, _ = topics.build_matrix(themed_tokens(10) + [["unseen"]], cfg)
    model = topics.train_lda(matrix, 3, cfg)
    assert topics.lda_assign(model, matrix)[-1] == -1


def test_run_lda_end_to_end_picks_three_topics():
    cfg = TopicConfig(no_below=2, workers=1, lda_passes=10, k_grid=(2, 3, 6), sweep_rows=180, coherence_rows=180, top_words=5)
    run = topics.run_lda(themed_tokens(), cfg)
    assert run.name == "lda" and run.n_topics == run.extras["chosen_k"] and set(run.extras["coherence_sweep"]) == {2, 3, 6}
    assert run.extras["chosen_k"] == 3 and run.coherence > 0.9 and len(run.assignments) == 180 and run.model is not None
    assert run.extras["coherence_sweep"][3] > run.extras["coherence_sweep"][2]


def test_bertopic_missing_gives_install_hint():
    with mock.patch.dict(sys.modules, {"bertopic": None}):
        with pytest.raises(ImportError, match="pip install bertopic"):
            topics.run_bertopic(make_frame(3), [], TopicConfig(), [])


def test_config_validation_and_smoke_copy():
    for bad in ({"sample_rows": 0}, {"k_grid": (1, 5)}, {"no_above": 0.0}, {"k_grid": ()}):
        with pytest.raises(ValueError):
            TopicConfig(**bad)
    smoke = TopicConfig().for_smoke()
    assert smoke.sample_rows == 15_000 and smoke.k_grid == (6, 10) and smoke.min_topic_size == 30


def test_topic_plots_render(tmp_path):
    table = topics.topic_table(make_frame(12), np.arange(12) % 3, {0: ["late", "box", "x", "y", "z"], 1: ["taste"], 2: ["price"]})
    p1 = plots.save_figure(plots.plot_coherence_sweep({6: 0.4, 8: 0.55}, 8, 0.5), tmp_path / "a.png", 50)
    p2 = plots.save_figure(plots.plot_topic_negativity(table, 0.1, "t"), tmp_path / "b.png", 50)
    assert p1.stat().st_size > 0 and p2.stat().st_size > 0


def fake_lda(tokens, cfg):
    n = len(tokens)
    return topics.TopicRun("lda", {0: ["late", "box"], 1: ["taste", "love"], 2: ["price", "cheap"]}, np.arange(n) % 3, 0.52,
                           {"coherence_sweep": {6: 0.4, 10: 0.52}, "chosen_k": 10}, {"fake": "model"})


def fake_bertopic(frame, tokens, cfg, drop):
    n = len(frame)
    return topics.TopicRun("bertopic", {0: ["refund", "late"], 1: ["taste", "flavor"]}, np.where(np.arange(n) % 5 == 0, -1, np.arange(n) % 2),
                           0.47, {"outlier_share": 0.2}, mock.Mock(save=mock.Mock(side_effect=RuntimeError("disk"))))


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(paths=PathsConfig(root=tmp_path), topics=TopicConfig(smoke_rows=60))


def test_run_requires_step3_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        run_topics.run(_cfg(tmp_path))


def test_run_end_to_end_with_fakes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()
    with mock.patch("reviews.run_topics.pd.read_parquet", return_value=make_frame(120)), \
            mock.patch.object(pd.DataFrame, "to_parquet", autospec=True) as to_parquet:
        summary = run_topics.run(cfg, sample=True, scope="all", lda_fn=fake_lda, bertopic_fn=fake_bertopic)
    assert summary["lda"]["meets_target"] is True and summary["bertopic"]["meets_target"] is False
    assert summary["reviews_used"] == 60 and summary["lda"]["n_topics"] == 3 and summary["bertopic"]["outlier_share"] == 0.2
    reports = cfg.paths.reports_path
    for name in ("topic_summary_sample.json", "topics_lda_sample.csv", "topics_bertopic_sample.csv"):
        assert (reports / name).is_file()
    assert len(list(cfg.paths.figures_path.glob("*_sample.png"))) == 3  # coherence sweep + two negativity charts
    saved = to_parquet.call_args[0][0]
    assert list(saved.columns) == ["Id", "lda_topic", "bertopic_topic"] and len(saved) == 60
    assert json.loads((reports / "topic_summary_sample.json").read_text())["scope"] == "all"


def test_run_negative_scope_changes_file_names(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()
    with mock.patch("reviews.run_topics.pd.read_parquet", return_value=make_frame(120)), \
            mock.patch.object(pd.DataFrame, "to_parquet", autospec=True):
        summary = run_topics.run(cfg, sample=False, scope="negative", lda_fn=fake_lda, bertopic_fn=fake_bertopic)
    assert summary["reviews_used"] == int(make_frame(120)["sentiment"].eq("negative").sum())
    assert (cfg.paths.reports_path / "topic_summary_negative.json").is_file()

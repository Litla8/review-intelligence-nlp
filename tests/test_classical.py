"""Step 5 tests on a small synthetic review set with a clear signal, so results are deterministic and fast."""
from __future__ import annotations

import json
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from reviews import classical, plots, train_classical
from reviews.config import ClassicalConfig, PathsConfig, ProjectConfig, SplitConfig

POS_WORDS = ["great", "love", "delicious", "perfect", "tasty"]
NEG_WORDS = ["bad", "awful", "stale", "waste", "horrible"]
FILLER = ["coffee", "tea", "box", "price", "flavor", "dog"]
FAST = ClassicalConfig(min_df=2, c_grid=(0.5, 1.0), alpha_grid=(0.5, 1.0), inference_sample=20, inference_repeats=1,
                       top_terms=3, error_rows=5, sample_rows=100)


def make_frame(n: int = 300, positive_share: float = 0.75, flip: int = 6) -> pd.DataFrame:
    """Deterministic reviews: positive ones use positive words, negative ones negative words; ``flip`` are mislabelled."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        positive = i % 4 != 0 if positive_share == 0.75 else rng.random() < positive_share
        words = list(rng.choice(POS_WORDS if positive else NEG_WORDS, 3)) + list(rng.choice(FILLER, 3))
        label = "positive" if positive else "negative"
        if i < flip:  # a few mislabelled reviews give the error table something to find
            label = "negative" if positive else "positive"
        rows.append({"Id": i, "sentiment": label, "Score": 5 if label == "positive" else 1, "Summary": "s",
                     "Text": " ".join(words).upper(), "text_clean": " ".join(words),
                     "review_date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=i)})
    frame = pd.DataFrame(rows)
    frame["sentiment"] = pd.Categorical(frame["sentiment"], categories=["negative", "neutral", "positive"])
    return frame


def test_binary_frame_drops_neutral_and_empty_text():
    frame = make_frame(8, flip=0)
    frame.loc[0, "sentiment"] = "neutral"
    frame.loc[1, "text_clean"] = ""
    out = classical.binary_frame(frame)
    assert len(out) == 6 and set(out["sentiment"]) <= {"negative", "positive"}
    assert out["sentiment"].map(type).eq(str).all()


def test_require_columns_and_vectorizer_kind():
    with pytest.raises(ValueError, match="Missing columns"):
        classical.binary_frame(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="Unknown feature kind"):
        classical.build_vectorizer("word2vec")
    assert type(classical.build_vectorizer("bow")).__name__ == "CountVectorizer"
    assert type(classical.build_vectorizer("tfidf")).__name__ == "TfidfVectorizer"


def test_stratified_split_sizes_overlap_and_ratio():
    data = classical.binary_frame(make_frame())
    train, val, test = classical.stratified_split(data, SplitConfig())
    assert len(train) + len(val) + len(test) == len(data)
    assert not (set(train.index) & set(val.index) or set(train.index) & set(test.index) or set(val.index) & set(test.index))
    shares = [(p["sentiment"] == "positive").mean() for p in (train, val, test)]
    assert max(shares) - min(shares) < 0.03
    assert abs(len(test) - 0.15 * len(data)) <= 1 and abs(len(val) - 0.15 * len(data)) <= 1


def test_stratified_split_is_reproducible_and_validates():
    data = classical.binary_frame(make_frame())
    a, _, _ = classical.stratified_split(data)
    b, _, _ = classical.stratified_split(data)
    assert list(a.index) == list(b.index)
    with pytest.raises(ValueError, match="two classes"):
        classical.stratified_split(data[data["sentiment"] == "positive"])


def test_chronological_split_orders_by_date():
    data = classical.binary_frame(make_frame()).sample(frac=1, random_state=1)  # shuffled input
    train, val, test = classical.chronological_split(data, SplitConfig())
    assert train["review_date"].max() <= val["review_date"].min() <= val["review_date"].max() <= test["review_date"].min()
    with pytest.raises(ValueError, match="Too few"):
        classical.chronological_split(data.head(3), SplitConfig())


def test_candidates_cover_four_models_with_imbalance_handling():
    cands = {c.name: c for c in classical.candidates(FAST)}
    assert set(cands) == {"bow_naive_bayes", "tfidf_naive_bayes", "tfidf_logreg", "tfidf_linear_svm"}
    assert cands["tfidf_logreg"].make(1.0).class_weight == "balanced"
    assert cands["tfidf_linear_svm"].make(1.0).class_weight == "balanced"
    assert cands["bow_naive_bayes"].make(1.0).fit_prior is False


@pytest.fixture(scope="module")
def experiment():
    data = classical.binary_frame(make_frame())
    train, val, test = classical.stratified_split(data)
    return data, test, classical.run_experiments(train, val, test, FAST, ["negative", "positive"])


def test_run_experiments_table_and_beats_baseline(experiment):
    _, _, result = experiment
    table = result["table"]
    assert set(table["model"]) == {"bow_naive_bayes", "tfidf_naive_bayes", "tfidf_logreg", "tfidf_linear_svm", "majority_class"}
    assert {"val_macro_f1", "test_macro_f1", "test_accuracy", "predict_seconds_per_1000"} <= set(table.columns)
    baseline = table.loc[table["model"] == "majority_class", "test_macro_f1"].iloc[0]
    best_f1 = table.loc[table["model"] == result["best"], "test_macro_f1"].iloc[0]
    assert baseline < 0.5 < best_f1 and best_f1 > 0.85
    assert result["best"] != "majority_class"
    assert table["test_macro_f1"].is_monotonic_decreasing


def test_run_experiments_is_deterministic(experiment):
    data, _, first = experiment
    train, val, test = classical.stratified_split(data)
    second = classical.run_experiments(train, val, test, FAST, ["negative", "positive"])
    cols = ["model", "best_param", "val_macro_f1", "test_macro_f1"]
    pd.testing.assert_frame_equal(first["table"][cols].sort_values("model").reset_index(drop=True),
                                  second["table"][cols].sort_values("model").reset_index(drop=True))


def test_tune_prefers_smaller_value_on_ties():
    cand = classical.Candidate("m", "tfidf", "C", (0.5, 5.0), lambda v: mock.Mock(fit=lambda x, y: mock.Mock(predict=lambda xv: np.array(["a", "b"]))))
    value, _, scores = classical.tune(cand, None, ["a", "b"], None, ["a", "b"], ["a", "b"])
    assert value == 0.5 and scores == {0.5: 1.0, 5.0: 1.0}


def test_parse_param():
    assert classical.parse_param("C=0.3") == ("C", 0.3)


def test_top_terms_signs(experiment):
    _, _, result = experiment
    pipeline = result["pipelines"]["tfidf_logreg"]
    terms = classical.top_terms(pipeline, 3)
    assert set(terms) == {"negative", "positive"}
    assert set(terms["positive"]) <= set(POS_WORDS) | set(FILLER) and set(terms["negative"]) <= set(NEG_WORDS) | set(FILLER)
    assert all(v > 0 for v in terms["positive"].values()) and all(v < 0 for v in terms["negative"].values())
    with pytest.raises(ValueError, match="coef_"):
        classical.top_terms(result["pipelines"]["bow_naive_bayes"], 3)


def test_confident_errors_only_mistakes(experiment):
    _, test, result = experiment
    errors = classical.confident_errors(result["pipelines"]["tfidf_logreg"], test, 5)
    assert len(errors) <= 5 and (errors["predicted"] != errors["sentiment"]).all()
    assert errors["confidence"].is_monotonic_decreasing and "Text" in errors.columns
    assert classical.confident_errors(result["pipelines"]["bow_naive_bayes"], test, 5).empty


def test_chronological_check_reports_dates(experiment):
    data, _, result = experiment
    row = result["table"].set_index("model").loc[result["best"]]
    check = classical.chronological_check(result["best"], row["best_param"], data, FAST, SplitConfig(), ["negative", "positive"])
    assert check["train_until"] < check["test_from"] and 0 <= check["macro_f1"] <= 1
    assert check["train_reviews"] + check["test_reviews"] == len(data)


def test_config_validation():
    for bad in ({"ngram_range": (2, 1)}, {"min_df": 0}, {"c_grid": ()}, {"alpha_grid": (0.0,)}):
        with pytest.raises(ValueError):
            ClassicalConfig(**bad)


def test_plots_render(experiment, tmp_path):
    _, _, result = experiment
    matrix = result["evaluations"][result["best"]]["confusion_matrix"]
    path = plots.save_figure(plots.plot_confusion_matrix(matrix, ["negative", "positive"], "t"), tmp_path / "cm.png", 50)
    path2 = plots.save_figure(plots.plot_model_comparison(result["table"]), tmp_path / "cmp.png", 50)
    assert path.stat().st_size > 0 and path2.stat().st_size > 0


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(paths=PathsConfig(root=tmp_path), classical=FAST)


def test_train_and_evaluate_and_write_outputs(tmp_path):
    cfg = _cfg(tmp_path)
    result = train_classical.train_and_evaluate(make_frame(), cfg)
    summary = result["summary"]
    json.dumps(summary)  # JSON-safe
    assert summary["best_test_macro_f1"] > 0.85 and "chronological_check" in summary and "top_terms" in summary
    paths = train_classical.write_outputs(result, cfg, sample=True)
    assert all(p.is_file() for p in paths.values()) and all("_sample" in p.name for p in paths.values())
    reloaded = joblib.load(paths["model"])
    texts = ["great love delicious", "bad awful stale"]
    assert list(reloaded.predict(texts)) == ["positive", "negative"]
    assert len(list((cfg.paths.figures_path).glob("*_sample.png"))) == 2


def test_run_requires_step3_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        train_classical.run(_cfg(tmp_path))


def test_run_with_mocked_dataset_load(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()  # only has to exist; the real read is mocked
    with mock.patch("reviews.train_classical.pd.read_parquet", return_value=make_frame()) as reader:
        summary = train_classical.run(cfg, sample=True)
    reader.assert_called_once()
    assert summary["reviews_used"] <= 100 and (cfg.paths.reports_path / "classical_results_sample.json").is_file()

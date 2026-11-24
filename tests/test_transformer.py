"""Step 6 tests. The GPU training function is replaced by a fake that returns logits, so no torch is needed."""
from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from reviews import train_transformer as tt
from reviews import transformer
from reviews.config import PathsConfig, ProjectConfig, TransformerConfig

CLASSES = ["negative", "positive"]


def make_frame(n: int = 200) -> pd.DataFrame:
    """Deterministic reviews, 3 of 4 positive, dated one per day so time slices are well defined."""
    labels = np.where(np.arange(n) % 4 == 0, "negative", "positive")
    return pd.DataFrame({
        "Id": np.arange(n), "sentiment": labels, "Score": np.where(labels == "positive", 5, 1), "Summary": "s",
        "Text": [f"Review number {i} text" for i in range(n)],
        "text_clean": [("great love" if lab == "positive" else "bad awful") + f" w{i % 7}" for i, lab in enumerate(labels)],
        "review_date": pd.Timestamp("2011-01-01") + pd.to_timedelta(np.arange(n), unit="D")})


def fake_logits(labels: pd.Series, noise: float = 0.0, seed: int = 42) -> np.ndarray:
    """Logits that favour the true class by 3, with optional noise, as a stand-in for a trained model."""
    rng = np.random.default_rng(seed)
    y = (labels.to_numpy() == "positive").astype(int)
    logits = np.zeros((len(y), 2))
    logits[np.arange(len(y)), y] = 3.0
    return logits + rng.normal(0, noise, logits.shape)


def fake_fit(train, val, test, cfg, model_dir):
    return tt.FitOutput(fake_logits(val["sentiment"], 1.5), fake_logits(test["sentiment"], 1.5, seed=7), 12.0,
                        lambda texts: np.zeros(len(texts), dtype=int))


def test_encoded_dataset_items_and_validation():
    ds = transformer.EncodedDataset({"input_ids": [[1, 2], [3]], "attention_mask": [[1, 1], [1]]}, [0, 1])
    assert len(ds) == 2 and ds[1] == {"input_ids": [3], "attention_mask": [1], "labels": 1}
    with pytest.raises(ValueError, match="one entry per label"):
        transformer.EncodedDataset({"input_ids": [[1]]}, [0, 1])


def test_label_ids_and_unknown():
    assert list(transformer.label_ids(["positive", "negative"], CLASSES)) == [1, 0]
    with pytest.raises(ValueError, match="Unknown"):
        transformer.label_ids(["neutral"], CLASSES)


def test_positive_probability_is_stable_and_valid():
    p = transformer.positive_probability(np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [-2.0, 2.0]]))
    assert np.allclose(p[:3], [0.5, 0.0, 1.0]) and 0.97 < p[3] < 0.99
    with pytest.raises(ValueError, match="shape"):
        transformer.positive_probability(np.zeros((3, 3)))


def test_compute_metrics():
    out = transformer.compute_metrics((np.array([[2, 0], [0, 2], [0, 2], [2, 0]]), np.array([0, 1, 0, 0])))
    assert out["accuracy"] == 0.75 and 0 < out["macro_f1"] < 1


def test_best_threshold_finds_the_separating_cut_and_prefers_near_half():
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([0.1] * 10 + [0.9] * 10)
    threshold, score = transformer.best_threshold(y, p, TransformerConfig())
    assert score == 1.0 and 0.1 < threshold <= 0.9 and abs(threshold - 0.5) < 0.05
    skewed = np.array([0.2] * 8 + [0.45] * 2 + [0.6] * 10)  # the two 0.45 reviews are negative; 0.5 splits cleanly
    assert transformer.best_threshold(y, skewed)[1] == 1.0


def test_apply_threshold_boundary_inclusive():
    assert list(transformer.apply_threshold(np.array([0.49, 0.5, 0.51]), 0.5)) == [0, 1, 1]


def test_slice_scores_use_dates_and_cover_both_ends():
    test = make_frame(90).sample(frac=1, random_state=3)  # shuffled, so ordering must come from the dates
    y = (test["sentiment"].to_numpy() == "positive").astype(int)
    perfect, wrong = y.copy(), 1 - y
    out = transformer.slice_scores(test, {"good": perfect, "bad": wrong}, y, 1 / 3)
    assert out["reviews_per_slice"] == 30 and out["oldest"]["to"] < out["newest"]["from"]
    assert out["oldest"]["macro_f1"]["good"] == 1.0 and out["newest"]["macro_f1"]["bad"] == 0.0
    with pytest.raises(ValueError, match="review_date"):
        transformer.slice_scores(test.drop(columns="review_date"), {}, y, 0.3)


def test_confident_errors_sorted_and_limited():
    test = make_frame(8)
    y = np.array([1, 1, 1, 0, 1, 1, 1, 0])
    pred = np.array([1, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.9, 0.2, 0.05, 0.1, 0.9, 0.9, 0.9, 0.7])
    out = transformer.confident_errors(test, p, pred, y, CLASSES, 5)
    assert list(out["Id"]) == [2, 1, 7] and list(out["predicted"]) == ["negative", "negative", "positive"]
    assert out["confidence"].is_monotonic_decreasing
    assert len(transformer.confident_errors(test, p, pred, y, CLASSES, 1)) == 1


def test_combine_tables_replaces_old_row_and_sorts():
    classical_table = pd.DataFrame({"model": ["svm", tt.MODEL_NAME], "test_macro_f1": [0.88, 0.1]})
    out = transformer.combine_tables(classical_table, {"model": tt.MODEL_NAME, "test_macro_f1": 0.93})
    assert list(out["model"]) == [tt.MODEL_NAME, "svm"] and len(out) == 2
    assert len(transformer.combine_tables(None, {"model": "x", "test_macro_f1": 1.0})) == 1


def test_gpu_message_gives_cu128_fix():
    assert transformer.gpu_message(True, "NVIDIA GeForce RTX 5070", "2.9.0") == "training on NVIDIA GeForce RTX 5070 (torch 2.9.0)"
    assert "cu128" in transformer.gpu_message(False, None, "2.9.0+cpu")


def test_transformer_config_validation():
    for bad in ({"epochs": 0}, {"learning_rate": 0.0}, {"warmup_fraction": 1.0}, {"threshold_min": 0.9, "threshold_max": 0.5},
                {"slice_fraction": 0.9}):
        with pytest.raises(ValueError):
            TransformerConfig(**bad)


def _split(frame):
    from reviews import classical
    return classical.stratified_split(classical.binary_frame(frame))


def test_assess_reports_verdict_slices_and_row():
    train, val, test = _split(make_frame())
    fit = fake_fit(train, val, test, TransformerConfig(), None)
    table = pd.DataFrame({"model": ["tfidf_linear_svm"], "test_macro_f1": [0.8821], "predict_seconds_per_1000": [0.03]})
    reference = lambda frame: np.where(frame["sentiment"] == "positive", "positive", "negative")
    out = tt.assess(fit, val, test, ProjectConfig(), CLASSES, table, 0.8821, reference)
    r = out["results"]
    assert r["gain_over_tfidf"] == round(r["tuned_threshold"]["macro_f1"] - 0.8821, 4)
    assert r["worth_the_cost"] == (r["gain_over_tfidf"] >= 0.01) and r["meets_target"] == (r["tuned_threshold"]["macro_f1"] > 0.88)
    assert set(r["time_slices"]["oldest"]["macro_f1"]) == {tt.MODEL_NAME, "tfidf_best"}
    assert out["row"]["tune_fit_seconds"] == 12.0 and tt.MODEL_NAME in set(out["table"]["model"])
    assert out["table"]["test_macro_f1"].is_monotonic_decreasing
    json.dumps(r)


def test_assess_without_baseline_skips_verdict_and_checks_lengths():
    train, val, test = _split(make_frame())
    fit = fake_fit(train, val, test, TransformerConfig(), None)
    r = tt.assess(fit, val, test, ProjectConfig(), CLASSES)["results"]
    assert "worth_the_cost" not in r and "tfidf_baseline_macro_f1" not in r
    bad = tt.FitOutput(fit.val_logits[:-1], fit.test_logits, 1.0, fit.predict)
    with pytest.raises(ValueError, match="different lengths"):
        tt.assess(bad, val, test, ProjectConfig(), CLASSES)


def test_label_array_accepts_names_and_ids():
    assert list(tt.label_array(["positive", "negative"], CLASSES)) == [1, 0]
    assert list(tt.label_array(np.array([1, 0]), CLASSES)) == [1, 0]


def _cfg(tmp_path) -> ProjectConfig:
    return ProjectConfig(paths=PathsConfig(root=tmp_path), transformer=TransformerConfig(inference_sample=20, inference_repeats=1))


def test_run_requires_step3_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        tt.run(_cfg(tmp_path))


def test_run_end_to_end_with_fake_training(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()  # only has to exist; the read is mocked
    with mock.patch("reviews.train_transformer.pd.read_parquet", return_value=make_frame(400)):
        results = tt.run(cfg, sample=True, fit_fn=fake_fit)
    reports = cfg.paths.reports_path
    assert (reports / "transformer_results_sample.json").is_file() and (reports / "model_comparison_all_sample.csv").is_file()
    assert (reports / "transformer_errors_sample.csv").is_file()
    assert len(list(cfg.paths.figures_path.glob("*_sample.png"))) == 2
    inference = json.loads((cfg.paths.models_path / "distilbert_sentiment_sample" / "inference_config.json").read_text())
    assert inference["threshold"] == results["threshold"] and inference["classes"] == CLASSES
    assert "worth_the_cost" not in results  # sample runs never compare with the full Step 5 run


def test_run_uses_full_step5_outputs_when_present(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.paths.processed_path.mkdir(parents=True)
    cfg.paths.text_path(False).touch()
    cfg.paths.reports_path.mkdir(parents=True)
    pd.DataFrame({"model": ["tfidf_linear_svm"], "test_macro_f1": [0.5], "predict_seconds_per_1000": [0.03]}).to_csv(
        cfg.paths.reports_path / "model_comparison.csv", index=False)
    (cfg.paths.reports_path / "classical_results.json").write_text(json.dumps({"tfidf_baseline_macro_f1_for_step6": 0.5}))
    with mock.patch("reviews.train_transformer.pd.read_parquet", return_value=make_frame(400)):
        results = tt.run(cfg, fit_fn=fake_fit)
    assert results["tfidf_baseline_macro_f1"] == 0.5 and "worth_the_cost" in results
    table = pd.read_csv(cfg.paths.reports_path / "model_comparison_all.csv")
    assert set(table["model"]) == {"tfidf_linear_svm", tt.MODEL_NAME}


def test_fit_distilbert_explains_missing_torch():
    import sys
    with mock.patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="cu128"):
            tt.fit_distilbert(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), TransformerConfig(), None)

"""Tests for reviews.metrics (tiny hard-coded data)."""
from __future__ import annotations

import pytest

from reviews.config import TargetConfig
from reviews.metrics import check_targets, evaluate_classifier, macro_f1, time_inference, worth_the_cost

TRUE = ["negative", "negative", "positive", "positive"]
PRED = ["negative", "positive", "positive", "positive"]


def test_macro_f1_known_value() -> None:
    # negative: p=1, r=0.5, f1=2/3; positive: p=2/3, r=1, f1=0.8 -> mean 0.7333
    assert macro_f1(TRUE, PRED) == pytest.approx((2 / 3 + 0.8) / 2)


def test_evaluate_classifier_report_and_missing_class() -> None:
    report = evaluate_classifier(TRUE, PRED, ["negative", "neutral", "positive"])
    assert report["accuracy"] == 0.75
    assert report["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [0, 0, 2]]
    assert report["per_class"]["neutral"]["support"] == 0 and report["per_class"]["neutral"]["f1"] == 0.0


def test_length_mismatch_and_empty_raise() -> None:
    with pytest.raises(ValueError):
        macro_f1(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        macro_f1([], [])


def test_time_inference_calls_predict_each_repeat() -> None:
    calls = []
    out = time_inference(lambda texts: calls.append(len(texts)), ["a"] * 10, repeats=3)
    assert calls == [10, 10, 10] and out["seconds_per_1000"] >= 0
    with pytest.raises(ValueError):
        time_inference(lambda t: None, [], repeats=1)


def test_targets_and_cost_rule() -> None:
    t = TargetConfig()
    assert check_targets(0.90, 0.55, t) == {"macro_f1_binary_above_target": True, "topic_coherence_above_target": True}
    assert check_targets(0.88, 0.5, t) == {"macro_f1_binary_above_target": False, "topic_coherence_above_target": False}
    assert worth_the_cost(0.90, 0.92, t) and not worth_the_cost(0.90, 0.905, t)

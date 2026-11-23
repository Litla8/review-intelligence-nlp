"""Step 5 models: TF-IDF / bag-of-words with Naive Bayes, Logistic Regression and Linear SVM.

The task is binary (negative vs positive; neutral is dropped, see ``labels.to_binary``). Everything here takes
DataFrames and a config, so it can be tested on a small synthetic frame and reused in notebooks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from reviews.config import ClassicalConfig, LabelConfig, SplitConfig
from reviews.metrics import evaluate_classifier, time_inference

LOGGER = logging.getLogger(__name__)
SEED = 42
TEXT = "text_clean"
LABEL = "sentiment"
BASELINE = "majority_class"


def require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a clear error if ``frame`` lacks any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def binary_frame(frame: pd.DataFrame, labels: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """Keep positive and negative reviews with non-empty cleaned text; ``sentiment`` becomes a plain string column."""
    require_columns(frame, [TEXT, LABEL])
    keep = frame[LABEL].astype(str).isin(labels.binary_classes) & frame[TEXT].fillna("").str.len().gt(0)
    out = frame.loc[keep].copy()
    out[LABEL] = out[LABEL].astype(str)
    LOGGER.info("binary task: %d of %d reviews kept", len(out), len(frame))
    return out


def stratified_split(frame: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test with the class ratio preserved in all three (random_state fixed)."""
    require_columns(frame, [LABEL])
    if frame[LABEL].nunique() < 2:
        raise ValueError("Need at least two classes to split")
    strat = frame[LABEL] if cfg.stratify else None
    rest, test = train_test_split(frame, test_size=cfg.test_size, stratify=strat, random_state=cfg.random_state)
    strat_rest = rest[LABEL] if cfg.stratify else None
    train, val = train_test_split(rest, test_size=cfg.val_size / (1 - cfg.test_size), stratify=strat_rest,
                                  random_state=cfg.random_state)
    return train, val, test


def chronological_split(frame: pd.DataFrame, cfg: SplitConfig = SplitConfig()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Oldest reviews for training, the next slice for validation, the newest for testing (no shuffling)."""
    require_columns(frame, ["review_date"])
    ordered = frame.sort_values("review_date", kind="stable")
    n = len(ordered)
    n_test, n_val = int(round(n * cfg.test_size)), int(round(n * cfg.val_size))
    if n_test < 1 or n_val < 1 or n - n_test - n_val < 1:
        raise ValueError("Too few reviews for a chronological split")
    return ordered.iloc[: n - n_val - n_test], ordered.iloc[n - n_val - n_test: n - n_test], ordered.iloc[n - n_test:]


def build_vectorizer(kind: str, cfg: ClassicalConfig = ClassicalConfig()) -> Any:
    """``bow`` (word counts) or ``tfidf``. ``text_clean`` is already tokenised, so split on whitespace only."""
    common = dict(token_pattern=r"\S+", lowercase=False, ngram_range=cfg.ngram_range, min_df=cfg.min_df,
                  max_features=cfg.max_features)
    if kind == "bow":
        return CountVectorizer(**common)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=cfg.sublinear_tf, **common)
    raise ValueError(f"Unknown feature kind: {kind!r} (use 'bow' or 'tfidf')")


@dataclass(frozen=True)
class Candidate:
    """One model family: which features it uses, how to build it for a parameter value, and the grid to try."""

    name: str
    features: str
    param: str
    grid: Tuple[float, ...]
    make: Callable[[float], Any]


def candidates(cfg: ClassicalConfig = ClassicalConfig()) -> List[Candidate]:
    """The four classical models. Class imbalance is handled with balanced class weights, or a uniform prior for NB."""
    return [
        Candidate("bow_naive_bayes", "bow", "alpha", cfg.alpha_grid, lambda a: MultinomialNB(alpha=a, fit_prior=False)),
        Candidate("tfidf_naive_bayes", "tfidf", "alpha", cfg.alpha_grid, lambda a: MultinomialNB(alpha=a, fit_prior=False)),
        Candidate("tfidf_logreg", "tfidf", "C", cfg.c_grid,
                  lambda c: LogisticRegression(C=c, class_weight="balanced", max_iter=cfg.max_iter, random_state=SEED)),
        Candidate("tfidf_linear_svm", "tfidf", "C", cfg.c_grid,
                  lambda c: LinearSVC(C=c, class_weight="balanced", random_state=SEED)),
    ]


def tune(candidate: Candidate, x_train: Any, y_train: Sequence[str], x_val: Any, y_val: Sequence[str],
         classes: Sequence[str]) -> Tuple[float, Any, Dict[float, float]]:
    """Fit once per grid value, score macro-F1 on the validation set, and return the best value, its model and all scores."""
    scores: Dict[float, float] = {}
    best_value, best_model, best_score = None, None, -1.0
    for value in candidate.grid:
        model = candidate.make(value).fit(x_train, y_train)
        score = evaluate_classifier(y_val, model.predict(x_val), classes)["macro_f1"]
        scores[value] = round(score, 4)
        if score > best_score:  # strict ">" keeps the smaller (more regularised) value on ties
            best_value, best_model, best_score = value, model, score
    return best_value, best_model, scores


def majority_baseline(y_train: Sequence[str], y_test: Sequence[str], classes: Sequence[str]) -> Dict[str, Any]:
    """What "always predict the common class" scores; every real model must beat this by a wide margin."""
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    return evaluate_classifier(y_test, dummy.predict(np.zeros((len(y_test), 1))), classes)


def run_experiments(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: ClassicalConfig,
                    classes: Sequence[str]) -> Dict[str, Any]:
    """Tune every candidate on validation, then score each once on the test set.

    Vectorisers are fitted on the training reviews only (no leakage) and shared by the candidates that use them.
    Returns ``table`` (one row per model, plus the baseline), ``pipelines``, ``evaluations`` and ``best`` (by validation macro-F1).
    """
    for part in (train, val, test):
        require_columns(part, [TEXT, LABEL])
    y_train, y_val, y_test = train[LABEL].to_numpy(), val[LABEL].to_numpy(), test[LABEL].to_numpy()
    vectorizers: Dict[str, Any] = {}
    matrices: Dict[str, Tuple[Any, Any, Any]] = {}
    pipelines: Dict[str, Pipeline] = {}
    evaluations: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    timing_texts = test[TEXT].head(cfg.inference_sample).tolist()
    for cand in candidates(cfg):
        if cand.features not in vectorizers:
            started = time.perf_counter()
            vec = build_vectorizer(cand.features, cfg)
            matrices[cand.features] = (vec.fit_transform(train[TEXT]), vec.transform(val[TEXT]), vec.transform(test[TEXT]))
            vectorizers[cand.features] = vec
            LOGGER.info("%s features: %d terms in %.0fs", cand.features, len(vec.vocabulary_), time.perf_counter() - started)
        x_train, x_val, x_test = matrices[cand.features]
        started = time.perf_counter()
        value, model, val_scores = tune(cand, x_train, y_train, x_val, y_val, classes)
        fit_seconds = time.perf_counter() - started
        pipeline = Pipeline([("features", vectorizers[cand.features]), ("model", model)])
        evaluation = evaluate_classifier(y_test, model.predict(x_test), classes)
        timing = time_inference(pipeline.predict, timing_texts, cfg.inference_repeats)
        pipelines[cand.name], evaluations[cand.name] = pipeline, evaluation
        rows.append({"model": cand.name, "features": cand.features, "best_param": f"{cand.param}={value}",
                     "val_macro_f1": val_scores[value], "test_accuracy": round(evaluation["accuracy"], 4),
                     "test_macro_f1": round(evaluation["macro_f1"], 4),
                     **{f"f1_{c}": round(evaluation["per_class"][c]["f1"], 4) for c in classes},
                     "tune_fit_seconds": round(fit_seconds, 1),
                     "predict_seconds_per_1000": round(timing["seconds_per_1000"], 3)})
        LOGGER.info("%s %s=%s: val macro-F1 %.4f, test macro-F1 %.4f", cand.name, cand.param, value, val_scores[value],
                    evaluation["macro_f1"])
    base = majority_baseline(y_train, y_test, classes)
    rows.append({"model": BASELINE, "features": "-", "best_param": "-", "val_macro_f1": np.nan,
                 "test_accuracy": round(base["accuracy"], 4), "test_macro_f1": round(base["macro_f1"], 4),
                 **{f"f1_{c}": round(base["per_class"][c]["f1"], 4) for c in classes},
                 "tune_fit_seconds": 0.0, "predict_seconds_per_1000": 0.0})
    table = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False, kind="stable").reset_index(drop=True)
    real = table[table["model"] != BASELINE]
    best = str(real.loc[real["val_macro_f1"].idxmax(), "model"])  # chosen on validation, never on test
    evaluations[BASELINE] = base
    return {"table": table, "pipelines": pipelines, "evaluations": evaluations, "best": best}


def parse_param(best_param: str) -> Tuple[str, float]:
    """Split ``"C=1.0"`` into ``("C", 1.0)``."""
    name, _, value = best_param.partition("=")
    return name, float(value)


def chronological_check(best: str, best_param: str, frame: pd.DataFrame, cfg: ClassicalConfig, split: SplitConfig,
                        classes: Sequence[str]) -> Dict[str, Any]:
    """Refit the chosen model with its chosen setting on the oldest reviews and test it on the newest.

    The stratified split mixes all years, so it can flatter a model when wording drifts. This is the harder, honest check.
    """
    cand = next(c for c in candidates(cfg) if c.name == best)
    _, value = parse_param(best_param)
    train, val, test = chronological_split(frame, split)
    older = pd.concat([train, val])  # tuning is already done, so validation reviews can join the training set
    vec = build_vectorizer(cand.features, cfg)
    pipeline = Pipeline([("features", vec), ("model", cand.make(value))]).fit(older[TEXT], older[LABEL])
    evaluation = evaluate_classifier(test[LABEL], pipeline.predict(test[TEXT]), classes)
    return {"model": best, "train_reviews": int(len(older)), "test_reviews": int(len(test)),
            "train_until": str(older["review_date"].max().date()) if "review_date" in older else None,
            "test_from": str(test["review_date"].min().date()), "test_positive_share": round(float((test[LABEL] == "positive").mean()), 4),
            "accuracy": round(evaluation["accuracy"], 4), "macro_f1": round(evaluation["macro_f1"], 4)}


def top_terms(pipeline: Pipeline, top_n: int) -> Dict[str, Dict[str, float]]:
    """Terms that push hardest towards each class, from a linear model's coefficients (``classes_[1]`` is positive)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        raise ValueError("top_terms needs a linear model with coef_")
    names = pipeline.named_steps["features"].get_feature_names_out()
    coef = np.asarray(model.coef_).ravel()
    order = np.argsort(coef)
    low_class, high_class = model.classes_[0], model.classes_[1]
    return {str(low_class): {str(names[i]): round(float(coef[i]), 3) for i in order[:top_n]},
            str(high_class): {str(names[i]): round(float(coef[i]), 3) for i in order[::-1][:top_n]}}


def confident_errors(pipeline: Pipeline, test: pd.DataFrame, limit: int) -> pd.DataFrame:
    """The ``limit`` mistakes the model was most sure about, with the original text, for reading by a human.

    Needs ``decision_function`` (Logistic Regression and Linear SVM); returns an empty frame for Naive Bayes.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "decision_function"):
        return pd.DataFrame()
    scores = pipeline.decision_function(test[TEXT])
    predicted = np.where(scores > 0, model.classes_[1], model.classes_[0])
    wrong = predicted != test[LABEL].to_numpy()
    out = test.loc[wrong, [c for c in ("Id", "Score", LABEL, "Summary", "Text") if c in test.columns]].copy()
    out["predicted"] = predicted[wrong]
    out["confidence"] = np.abs(scores[wrong]).round(3)
    return out.sort_values("confidence", ascending=False, kind="stable").head(limit).reset_index(drop=True)

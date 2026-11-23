"""Tests for reviews.text and reviews.preprocess. spaCy is replaced by a tiny fake pipeline."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from reviews.config import ProjectConfig, TextConfig
from reviews.data import clean_reviews
from reviews.preprocess import main, run
from reviews.text import (
    clean_text, expand_contractions, filter_tokens, lemmatize, load_spacy, preprocess, stopword_set, strip_markup,
    text_report)

CFG = TextConfig()
LEMMAS = {"was": "be", "bars": "bar", "loved": "love", "tastes": "taste", "were": "be", "dogs": "dog"}


class FakeNlp:
    """Looks like a spaCy pipeline: ``pipe`` yields docs whose tokens have ``lemma_`` and ``is_space``."""

    def __init__(self) -> None:
        self.calls = []

    def pipe(self, texts, batch_size, n_process):
        self.calls.append({"n": len(texts), "batch_size": batch_size, "n_process": n_process})
        for text in texts:
            yield [SimpleNamespace(lemma_=LEMMAS.get(w, w), is_space=False) for w in text.split()]


def test_strip_markup_removes_tags_and_entities() -> None:
    assert strip_markup("Great<br />taste &amp; smell").split() == ["Great", "taste", "&", "smell"]


@pytest.mark.parametrize("raw, expected", [
    ("I don't like it", "I do not like it"),
    ("It won't work", "It will not work"),
    ("I can't wait", "I cannot wait"),
    ("They're great, I'm happy, we've tried", "They are great, I am happy, we have tried"),
    ("It\u2019s fine, ain\u2019t it", "It fine, is not it"),
])
def test_expand_contractions(raw: str, expected: str) -> None:
    assert expand_contractions(raw) == expected


def test_clean_text_full_pipeline() -> None:
    raw = "Loved it!!! <br />Visit http://x.com or www.y.org :) Paid $12.99 for 3 bars; don't buy again :("
    assert clean_text(raw) == "loved it visit or happyface paid for bars do not buy again sadface"


def test_clean_text_exact_small_cases() -> None:
    assert clean_text("Great!!  Taste.") == "great taste"
    assert clean_text("NOT good") == "not good"
    assert clean_text("5 stars, 100%") == "stars"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_switches_turn_steps_off() -> None:
    assert clean_text("3 bars", TextConfig(remove_digits=False)) == "3 bars"
    assert clean_text("Nice <b>Tea</b>", TextConfig(lowercase=False)) == "Nice Tea"
    assert "br" in clean_text("a<br/>b", TextConfig(remove_html=False))
    assert "http" in clean_text("see http://x.com now", TextConfig(remove_urls=False))


def test_stopwords_keep_negations_and_intensifiers() -> None:
    stop = stopword_set(CFG)
    assert "the" in stop and "and" in stop
    assert not ({"not", "no", "never", "but", "very", "too", "nothing"} & stop)
    tokens = filter_tokens("the taste was not very good but a bit too sweet".split(), stop, CFG)
    assert tokens == ["taste", "not", "very", "good", "but", "bit", "too", "sweet"]


def test_short_tokens_are_dropped() -> None:
    assert filter_tokens(["x", "ok", "fine"], frozenset(), CFG) == ["ok", "fine"]


def test_lemmatize_uses_workers_only_for_large_batches() -> None:
    nlp = FakeNlp()
    out = lemmatize(["the bars were good", "dogs loved it"], nlp, CFG)
    assert out == [["the", "bar", "be", "good"], ["dog", "love", "it"]]
    assert nlp.calls[0]["n_process"] == 1                       # 2 texts < batch_size
    lemmatize(["a"] * CFG.batch_size, nlp, CFG)
    assert nlp.calls[1]["n_process"] == CFG.n_process


def test_preprocess_adds_columns_and_keeps_raw_text() -> None:
    frame = pd.DataFrame({"Text": ["The bars were NOT good!", "Dogs loved the tastes <br/>", None, "the and of"],
                          "n_words": [5, 4, 0, 3]}, index=[10, 11, 12, 13])
    out = preprocess(frame, FakeNlp(), CFG)
    assert out["text_clean"].tolist() == ["bar not good", "dog love taste", "", ""]
    assert out["n_tokens"].tolist() == [3, 3, 0, 0] and out["n_tokens"].dtype == "int32"
    assert out["Text"].tolist() == frame["Text"].tolist() and out.index.tolist() == [10, 11, 12, 13]
    assert "text_clean" not in frame.columns                    # the input frame is not modified
    with pytest.raises(ValueError, match="Text"):
        preprocess(frame.drop(columns=["Text"]), FakeNlp(), CFG)


def test_text_report_checks_negations_and_empties() -> None:
    frame = pd.DataFrame({"text_clean": ["not good bar", "good bar", "", "never again bar"],
                          "n_tokens": [3, 2, 0, 3], "n_words": [6, 4, 2, 5]})
    report = text_report(frame, CFG)
    assert report["rows"] == 4 and report["empty_after_cleaning"] == 1
    assert report["kept_sentiment_words"]["not"] == 1 and report["kept_sentiment_words"]["never"] == 1
    assert report["top_tokens"]["bar"] == 3 and report["vocabulary_size"] == 5
    assert report["reviews_with_negation_share"] == 0.5
    json.dumps(report)


def test_load_spacy_explains_how_to_install() -> None:
    with mock.patch.dict("sys.modules", {"spacy": None}):
        with pytest.raises(ImportError, match="spacy download"):
            load_spacy(CFG)


def test_load_spacy_disables_parser_and_ner() -> None:
    fake = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"spacy": fake}):
        load_spacy(CFG)
    fake.load.assert_called_once_with("en_core_web_sm", disable=["parser", "ner"])


def test_run_requires_step_2_output(cfg: ProjectConfig) -> None:
    with pytest.raises(FileNotFoundError, match="reviews.prepare"):
        run(cfg, nlp_loader=lambda _cfg: FakeNlp())


def _write_clean(cfg: ProjectConfig, raw_reviews: pd.DataFrame) -> None:
    cfg.paths.processed_path.mkdir(parents=True, exist_ok=True)
    clean_reviews(raw_reviews, cfg.data).frame.to_parquet(cfg.paths.clean_path, index=False)


def test_run_writes_outputs_full_and_sample(cfg: ProjectConfig, raw_reviews: pd.DataFrame) -> None:
    pytest.importorskip("pyarrow")
    _write_clean(cfg, raw_reviews)
    loader = lambda _cfg: FakeNlp()  # noqa: E731
    report = run(cfg, nlp_loader=loader)
    out = pd.read_parquet(cfg.paths.text_path())
    assert len(out) == 10 and {"text_clean", "n_tokens", "Text", "sentiment"} <= set(out.columns)
    assert json.loads(cfg.paths.text_report_path.read_text(encoding="utf-8")) == report
    assert run(cfg, nlp_loader=loader) == report                     # repeatable
    small = ProjectConfig(paths=cfg.paths, text=TextConfig(sample_rows=4))
    sample_report = run(small, sample=True, nlp_loader=loader)
    assert sample_report["rows"] == 4 and cfg.paths.text_path(sample=True).is_file()
    again = pd.read_parquet(cfg.paths.text_path(sample=True))
    assert run(small, sample=True, nlp_loader=loader)["rows"] == 4
    pd.testing.assert_frame_equal(again, pd.read_parquet(cfg.paths.text_path(sample=True)))  # same rows every time


def test_main_passes_options(cfg: ProjectConfig, raw_reviews: pd.DataFrame) -> None:
    pytest.importorskip("pyarrow")
    _write_clean(cfg, raw_reviews)
    with mock.patch("reviews.preprocess.load_spacy", return_value=FakeNlp()):
        assert main(["--root", str(cfg.paths.root), "--sample", "--n-process", "1", "--log-level", "WARNING"]) == 0
    assert cfg.paths.text_path(sample=True).is_file()


def test_text_config_validation() -> None:
    with pytest.raises(ValueError):
        TextConfig(n_process=0)

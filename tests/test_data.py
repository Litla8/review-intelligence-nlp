"""Tests for reviews.data (tiny deterministic frames, no real files)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reviews.config import DataConfig
from reviews.data import (
    add_helpfulness_ratio, add_labels, clean_reviews, coerce_types, drop_duplicate_reviews, drop_duplicate_texts,
    drop_invalid_ratings, drop_non_english, drop_unusable_text, english_mask, load_raw, normalise_text,
    validate_columns)

CFG = DataConfig()


def test_validate_columns_names_the_missing_ones(raw_reviews: pd.DataFrame) -> None:
    validate_columns(raw_reviews, CFG)
    with pytest.raises(ValueError, match="Text"):
        validate_columns(raw_reviews.drop(columns=["Text"]), CFG)


def test_load_raw_missing_file_says_where_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_raw(tmp_path / "Reviews.csv")


def test_load_raw_reads_a_csv(tmp_path: Path, raw_reviews: pd.DataFrame) -> None:
    path = tmp_path / "Reviews.csv"
    raw_reviews.to_csv(path, index=False)
    loaded = load_raw(path)
    assert len(loaded) == len(raw_reviews) and loaded["HelpfulnessNumerator"].dtype == "float64"


def test_coerce_types_makes_dates_and_numeric_ratings(raw_reviews: pd.DataFrame) -> None:
    frame = raw_reviews.assign(Score=raw_reviews["Score"].astype("object"))
    frame.loc[0, "Score"] = "5"
    out = coerce_types(frame, CFG)
    assert out["Score"].dtype.kind == "f" and out.loc[0, "Score"] == 5.0
    assert out["review_date"].iloc[0] == pd.Timestamp(1_300_000_000 + 86_400, unit="s")  # row 1 is posted one day after T0


def test_drop_unusable_text_removes_empty_missing_and_short(raw_reviews: pd.DataFrame) -> None:
    kept = drop_unusable_text(raw_reviews, CFG)
    assert set(raw_reviews["Id"]) - set(kept["Id"]) == {13, 14, 15}


def test_drop_invalid_ratings(raw_reviews: pd.DataFrame) -> None:
    kept = drop_invalid_ratings(coerce_types(raw_reviews, CFG), CFG)
    assert set(raw_reviews["Id"]) - set(kept["Id"]) == {16, 17}
    half = coerce_types(raw_reviews.assign(Score=[4.5] + [5] * 18), CFG)  # a half star is not a rating
    assert 1 not in set(drop_invalid_ratings(half, CFG)["Id"])


def test_exact_duplicate_is_dropped_keeping_the_first(raw_reviews: pd.DataFrame) -> None:
    kept = drop_duplicate_reviews(raw_reviews, CFG)
    assert 1 in set(kept["Id"]) and 11 not in set(kept["Id"]) and 12 in set(kept["Id"])


def test_text_copies_are_found_across_products_and_case(raw_reviews: pd.DataFrame) -> None:
    kept = drop_duplicate_texts(raw_reviews, CFG)
    assert 2 in set(kept["Id"]) and 12 not in set(kept["Id"]) and 11 not in set(kept["Id"])
    off = DataConfig(drop_duplicate_texts=False)
    assert len(drop_duplicate_texts(raw_reviews, off)) == len(raw_reviews)


def test_normalise_text_only_for_matching() -> None:
    assert normalise_text(pd.Series(["  A  B\n c "])).tolist() == ["a b c"]


def test_english_mask_keeps_english_and_short_ascii_drops_other_languages(raw_reviews: pd.DataFrame) -> None:
    mask = english_mask(raw_reviews["Text"], CFG)
    dropped = set(raw_reviews.loc[~mask, "Id"])
    assert {18, 19} <= dropped and not ({1, 2, 9, 10} & dropped)


def test_english_filter_can_be_switched_off(raw_reviews: pd.DataFrame) -> None:
    off = DataConfig(english_only=False)
    assert len(drop_non_english(raw_reviews, off)) == len(raw_reviews)


def test_helpfulness_ratio_marks_impossible_votes(raw_with_votes: pd.DataFrame) -> None:
    out, invalid = add_helpfulness_ratio(raw_with_votes, CFG)
    assert invalid == 1
    assert pd.isna(out["helpful_ratio"].iloc[0]) and pd.isna(out["helpful_ratio"].iloc[1])
    assert out["helpful_ratio"].iloc[2] == pytest.approx(0.75)


def test_add_labels_and_word_counts(raw_reviews: pd.DataFrame) -> None:
    out = add_labels(raw_reviews.iloc[:3].copy(), CFG)
    assert out["sentiment"].astype(str).tolist() == ["positive", "negative", "neutral"]
    assert out["n_words"].tolist()[0] == len(raw_reviews["Text"].iloc[0].split())


def test_clean_reviews_end_to_end_keeps_exactly_the_good_rows(raw_reviews: pd.DataFrame) -> None:
    result = clean_reviews(raw_reviews, CFG)
    assert result.frame["Id"].tolist() == list(range(1, 11))
    dropped = {s["step"]: s["dropped"] for s in result.stages}
    assert dropped == {"unusable_text": 3, "invalid_rating_or_date": 2, "duplicate_review": 1,
                       "duplicate_text": 1, "non_english": 2}
    assert result.stages[-1]["remaining"] == 10
    assert list(result.frame.columns)[:6] == ["Id", "ProductId", "UserId", "review_date", "Score", "sentiment"]
    assert result.frame["Text"].iloc[0] == raw_reviews["Text"].iloc[0]  # raw text is never altered


def test_clean_reviews_is_idempotent(raw_reviews: pd.DataFrame) -> None:
    first = clean_reviews(raw_reviews, CFG).frame
    second = clean_reviews(raw_reviews, CFG).frame
    pd.testing.assert_frame_equal(first, second)


def test_clean_reviews_raises_when_nothing_survives(raw_reviews: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No reviews survived"):
        clean_reviews(raw_reviews.assign(Text=""), CFG)


def test_config_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        DataConfig(min_ascii_share=1.5)
    with pytest.raises(ValueError):
        DataConfig(min_words=0)
    with pytest.raises(ValueError):
        DataConfig(duplicate_key=())

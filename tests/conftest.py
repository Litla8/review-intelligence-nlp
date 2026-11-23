"""Shared deterministic fixtures: a tiny review file with one example of every problem the cleaner handles."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reviews.config import ProjectConfig

COLUMNS = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score",
           "Time", "Summary", "Text"]
DAY = 86_400
T0 = 1_300_000_000  # March 2011


def _row(i: int, product: str, user: str, score, text, yes=1, total=2, time=None, summary="Summary"):
    return [i, product, user, f"user{i}", yes, total, score, T0 + i * DAY if time is None else time, summary, text]


def build_rows() -> list:
    """Ten good reviews plus one row per defect; the expected survivors are the first ten."""
    good = [
        _row(1, "P1", "U1", 5, "This coffee is great and I love the taste of it"),
        _row(2, "P1", "U2", 1, "The package was broken and the food was not fresh at all"),
        _row(3, "P2", "U3", 3, "It is okay but the price is a little high for what you get"),
        _row(4, "P2", "U4", 4, "My dog likes this treat and it is made with real chicken"),
        _row(5, "P3", "U5", 2, "I did not like the flavor and the texture was strange to me"),
        _row(6, "P3", "U6", 5, "We have bought this tea many times and it is always the best"),
        _row(7, "P4", "U7", 5, "Fast delivery and the box was in perfect shape when it arrived"),
        _row(8, "P4", "U8", 1, "This was stale and I had to throw it away after one bite"),
        _row(9, "P5", "U9", 4, "Great taste overall"),                                    # short English text: only the ASCII test applies
        _row(10, "P5", "U10", 5, "A very good snack for kids and it is not too sweet at all"),
    ]
    defects = [
        _row(11, "P1", "U1", 5, "This coffee is great and I love the taste of it", time=T0 + 1 * DAY),  # exact duplicate of 1
        _row(12, "P9", "U99", 1, "  THE package was broken and the food was NOT fresh at all  "),         # same words as 2
        _row(13, "P6", "U13", 5, ""),                                                                    # empty
        _row(14, "P6", "U14", 5, None),                                                                  # missing
        _row(15, "P6", "U15", 5, "ok"),                                                                  # too short
        _row(16, "P6", "U16", 7, "A normal long enough review with the words that I like it"),           # rating 7
        _row(17, "P6", "U17", None, "Another normal long enough review with the words that I like"),     # no rating
        _row(18, "P6", "U18", 5, "Este producto es muy bueno y lo recomiendo con todo mi corazon amigos"),  # Spanish
        _row(19, "P6", "U19", 5, "Очень вкусный чай и отличный аромат для всей нашей семьи"),                           # Russian (non-ASCII)
    ]
    return good + defects


@pytest.fixture
def raw_reviews() -> pd.DataFrame:
    return pd.DataFrame(build_rows(), columns=COLUMNS)


@pytest.fixture
def raw_with_votes() -> pd.DataFrame:
    """Two clean reviews, one with impossible votes and one with no votes at all."""
    rows = [_row(1, "P1", "U1", 5, "This coffee is great and I love the taste of it", yes=5, total=2),
            _row(2, "P2", "U2", 1, "The package was broken and the food was not fresh at all", yes=0, total=0),
            _row(3, "P3", "U3", 4, "My dog likes this treat and it is made with real chicken", yes=3, total=4)]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def cfg(tmp_path: Path) -> ProjectConfig:
    from dataclasses import replace
    base = ProjectConfig()
    return replace(base, paths=replace(base.paths, root=tmp_path))

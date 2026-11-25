"""Tests for reviews.summarize (the model is replaced by a fake)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import pytest

from reviews import summarize as sm
from reviews.config import ProjectConfig, SummaryConfig


def count(text: str) -> int:
    return len(text.split())


def fake_summarize(chunks: Sequence[str]) -> List[str]:
    return [" ".join(c.split()[:5]) for c in chunks]    # first five words: shrinks every chunk


def test_clean_for_summary() -> None:
    assert sm.clean_for_summary("Good <br />  coffee\n here") == "Good coffee here"


def test_pack_chunks_respects_limit() -> None:
    chunks = sm.pack_chunks(["a b c", "d e f", "g h i"], count, 6)
    assert chunks == ["a b c d e f", "g h i"]
    assert sm.pack_chunks(["a b c d e f g"], count, 3) == ["a b c d e f g"]    # over-long text stays whole
    assert sm.pack_chunks([], count, 3) == []


def test_summarize_texts_map_reduce() -> None:
    cfg = SummaryConfig(max_input_tokens=12, batch_size=2)
    texts = [f"review number {i} says the coffee was stale and bitter" for i in range(6)]
    out = sm.summarize_texts(texts, fake_summarize, count, cfg)
    assert out["levels"] >= 2 and out["chunks"] > 1 and out["summary"]


def test_summarize_texts_empty_raises() -> None:
    with pytest.raises(ValueError):
        sm.summarize_texts(["", "  "], fake_summarize, count)


def _reviews(n: int = 60) -> pd.DataFrame:
    base = "the shipping was slow and the box arrived damaged with the coffee beans stale and bitter overall "
    return pd.DataFrame({"Id": range(n), "Text": [base * 2 + f"batch {i}" for i in range(n)],
                         "text_clean": [base * 2 + f"batch {i}" for i in range(n)], "sentiment": ["negative"] * n})


def test_representative_positions_deterministic_and_filtered() -> None:
    frame = _reviews()
    frame.loc[0, "text_clean"] = "short"
    a = sm.representative_positions(frame["text_clean"], 5, 10, 150, 1000)
    b = sm.representative_positions(frame["text_clean"], 5, 10, 150, 1000)
    assert a == b and len(a) == 5 and 0 not in a
    assert sm.representative_positions(pd.Series(["tiny"]), 5, 10, 150, 10) == []


def test_summarize_pain_points_skips_small_groups() -> None:
    cfg = replace(ProjectConfig(), summary=replace(SummaryConfig(), min_group_reviews=10, reviews_per_group=5))
    res = sm.summarize_pain_points(_reviews(), cfg, fake_summarize, count)
    assert "summary" in res["overall"] and "summary" in res["shipping"]
    assert "skipped" in res["price"] and res["price"]["negative_reviews"] == 0


def test_run_writes_reports(tmp_path: Path) -> None:
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=tmp_path))
    source = cfg.paths.text_path(sample=False)
    source.parent.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_parquet", lambda *a, **k: _reviews())
        with pytest.raises(FileNotFoundError):
            sm.run(cfg)
        source.write_text("x")
        res = sm.run(cfg, sample=True, loader=lambda c: (fake_summarize, count))
    assert "overall" in res
    assert (cfg.paths.reports_path / "pain_point_summaries_sample.json").is_file()
    assert "Pain-point summaries" in (cfg.paths.reports_path / "pain_point_summaries_sample.md").read_text()


def test_summary_config_validation() -> None:
    with pytest.raises(ValueError):
        SummaryConfig(min_new_tokens=100, max_new_tokens=10)

"""Step 4 charts. Each function returns a matplotlib Figure; :func:`save_figure` writes and closes it.

The Agg backend is forced because these run in scripts and tests, not an interactive window.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after the backend is chosen)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

LOGGER = logging.getLogger(__name__)
# One colour per sentiment, used on every chart. Blue and orange stay distinguishable for colour-blind readers.
SENTIMENT_COLOURS: Dict[str, str] = {"negative": "#D55E00", "neutral": "#8C8C8C", "positive": "#0072B2"}
ORDER = ("negative", "neutral", "positive")


def save_figure(fig: Figure, path: Path, dpi: int = 150) -> Path:
    """Write the figure to ``path`` (creating folders) and close it so memory is released."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_length_distribution(frame: pd.DataFrame, clip_quantile: float = 0.99) -> Figure:
    """Overlaid histograms of review length in words, one per sentiment, cut at ``clip_quantile``."""
    upper = float(frame["n_words"].quantile(clip_quantile))
    bins = np.linspace(0, max(upper, 1.0), 41)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ORDER:
        subset = frame.loc[frame["sentiment"].astype(str) == name, "n_words"]
        if len(subset):
            ax.hist(subset.clip(upper=upper), bins=bins, density=True, histtype="step", linewidth=2,
                    color=SENTIMENT_COLOURS[name], label=f"{name} (median {subset.median():.0f})")
    ax.set_xlabel("Words per review")
    ax.set_ylabel("Share of reviews")
    ax.set_title("Review length by sentiment")
    ax.legend(frameon=False)
    return fig


def _bar_grid(series_by_class: Mapping[str, pd.Series], title: str) -> Figure:
    """One horizontal bar chart per sentiment, side by side, largest at the top."""
    names = [n for n in ORDER if n in series_by_class]
    fig, axes = plt.subplots(1, max(len(names), 1), figsize=(5.2 * max(len(names), 1), 6), squeeze=False)
    for ax, name in zip(axes[0], names):
        data = series_by_class[name].iloc[::-1]
        ax.barh(data.index, data.to_numpy(), color=SENTIMENT_COLOURS[name])
        ax.set_title(name.capitalize())
        ax.set_xlabel("Count")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_top_words(words_by_class: Mapping[str, pd.Series]) -> Figure:
    """Most frequent content words for each sentiment."""
    return _bar_grid(words_by_class, "Most frequent words by sentiment")


def plot_ngrams(ngrams_by_class: Mapping[str, pd.Series], label: str) -> Figure:
    """Most frequent phrases (``label`` is "Bigrams" or "Trigrams") for each sentiment."""
    return _bar_grid(ngrams_by_class, f"Most frequent {label.lower()} by sentiment")


def plot_distinctive(negative: pd.Series, positive: pd.Series) -> Figure:
    """Words most typical of negative reviews (left) and of positive reviews (right), by log-odds z-score."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, data, name in ((axes[0], negative, "negative"), (axes[1], positive, "positive")):
        data = data.iloc[::-1]
        ax.barh(data.index, data.abs().to_numpy(), color=SENTIMENT_COLOURS[name])
        ax.set_title(f"Typical of {name} reviews")
        ax.set_xlabel("Distinctiveness (z-score)")
    fig.tight_layout()
    return fig


def plot_rating_trend(monthly: pd.DataFrame) -> Figure:
    """Share of positive and negative reviews per month (top) and review volume (bottom)."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[2, 1])
    top.plot(monthly.index, monthly["share_positive"], color=SENTIMENT_COLOURS["positive"], linewidth=2,
             label="Positive (4-5 stars)")
    top.plot(monthly.index, monthly["share_negative"], color=SENTIMENT_COLOURS["negative"], linewidth=2,
             label="Negative (1-2 stars)")
    top.set_ylim(0, 1)
    top.set_ylabel("Share of reviews")
    top.set_title("Sentiment over time")
    top.legend(frameon=False, loc="center left")
    bottom.bar(monthly.index, monthly["n_reviews"], width=25, color="#8C8C8C")
    bottom.set_ylabel("Reviews per month")
    fig.tight_layout()
    return fig


def _default_cloud(max_words: int) -> Any:
    """Build a WordCloud, with a friendly error if the package is not installed."""
    try:
        from wordcloud import WordCloud
    except ImportError as exc:
        raise ImportError("wordcloud is not installed. Run: pip install wordcloud") from exc
    return WordCloud(width=900, height=500, background_color="white", max_words=max_words, random_state=42)


def plot_wordclouds(counts_by_class: Mapping[str, pd.Series], max_words: int, cloud: Optional[Any] = None) -> Figure:
    """One word cloud per sentiment. ``cloud`` can be any object with ``generate_from_frequencies`` and ``to_array``."""
    names = [n for n in ORDER if n in counts_by_class]
    fig, axes = plt.subplots(1, max(len(names), 1), figsize=(6 * max(len(names), 1), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        generator = cloud if cloud is not None else _default_cloud(max_words)
        image = generator.generate_from_frequencies(counts_by_class[name].head(max_words * 2).to_dict()).to_array()
        ax.imshow(image, interpolation="bilinear")
        ax.set_title(name.capitalize(), color=SENTIMENT_COLOURS[name])
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix: Any, classes: Any, title: str) -> Figure:
    """Confusion matrix with row-normalised colours (share of each true class) and raw counts written in the cells."""
    counts = np.asarray(matrix, dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), labels=list(classes))
    ax.set_yticks(range(len(classes)), labels=list(classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(counts[i, j]):,}\n{shares[i, j]:.0%}", ha="center", va="center",
                    color="white" if shares[i, j] > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, label="Share of actual class")
    fig.tight_layout()
    return fig


def plot_model_comparison(table: pd.DataFrame) -> Figure:
    """Test macro-F1 per model (left) and prediction time per 1,000 reviews (right)."""
    ordered = table.sort_values("test_macro_f1")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.barh(ordered["model"], ordered["test_macro_f1"], color=SENTIMENT_COLOURS["positive"])
    left.set_xlim(0, 1)
    left.set_xlabel("Test macro-F1 (higher is better)")
    for y, value in enumerate(ordered["test_macro_f1"]):
        left.text(value + 0.01, y, f"{value:.3f}", va="center")
    timed = ordered[ordered["predict_seconds_per_1000"] > 0]
    right.barh(timed["model"], timed["predict_seconds_per_1000"], color="#8C8C8C")
    right.set_xlabel("Seconds to predict 1,000 reviews (lower is better)")
    fig.tight_layout()
    return fig

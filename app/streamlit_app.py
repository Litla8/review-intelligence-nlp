"""Streamlit app: paste a review, or upload a CSV for an insights dashboard.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # so `reviews` imports work however streamlit is launched
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from reviews import analyzer, predict, summarize
from reviews.config import ProjectConfig

CFG = ProjectConfig()
LOCAL_MODEL_DIR = ROOT / CFG.paths.models_dir / CFG.transformer.model_dir
MODEL_REPO = os.environ.get("HF_MODEL_REPO")            # set as a secret when deployed
SUMMARIES = os.environ.get("ENABLE_SUMMARY", "1" if LOCAL_MODEL_DIR.is_dir() else "0") == "1"   # off on small hosts


@st.cache_resource(show_spinner="Loading the sentiment model...")
def get_scorers() -> analyzer.Scorers:
    """Load DistilBERT once (GPU if present, else CPU) and share it across sessions."""
    model_dir = predict.resolve_model_dir(LOCAL_MODEL_DIR, MODEL_REPO)
    review, sentence = predict.load_scorers(model_dir, [(CFG.app.review_max_length, CFG.app.batch_size),
                                                        (CFG.aspects.max_length, CFG.aspects.batch_size)], require_cuda=False)
    return analyzer.Scorers(review, sentence, predict.read_threshold(model_dir))


def app_settings():
    """Smaller upload cap on CPU-only hosts (override with the APP_MAX_ROWS environment variable / secret)."""
    default = CFG.app.max_rows if predict.cuda_available() else 2_000
    return replace(CFG.app, max_rows=int(os.environ.get("APP_MAX_ROWS", default)))


@st.cache_resource(show_spinner="Loading the summarisation model (first use downloads it)...")
def get_summarizer():
    """Load the BART summariser once."""
    return summarize.load_summarizer(CFG.summary)


def single_review_tab(scorers: analyzer.Scorers) -> None:
    """Paste one review: overall sentiment, aspect sentences and an optional summary."""
    text = st.text_area("Paste a review", height=180, placeholder="The coffee tasted great but the box arrived crushed...")
    if not st.button("Analyse review", type="primary"):
        return
    try:
        result = analyzer.analyze_review(text, scorers, CFG.aspects)
    except ValueError as exc:
        st.warning(str(exc))
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", result["label"].capitalize())
    col2.metric("P(positive)", f"{result['p_positive']:.1%}")
    col3.metric("Words", result["words"])
    if not result["confident"]:
        st.info("The model is not confident either way, so this review is shown as mixed.")
    if result["aspect_sentences"]:
        st.subheader("Aspects mentioned")
        st.dataframe(pd.DataFrame(result["aspect_sentences"]), hide_index=True, use_container_width=True)
    else:
        st.caption("No aspect keywords (taste, price, shipping...) found in this review.")
    if SUMMARIES and result["words"] >= CFG.app.long_review_words:
        summarizer, counter = get_summarizer()
        st.subheader("Summary")
        st.write(summarize.summarize_texts([text], summarizer, counter, CFG.summary)["summary"])


def dashboard_tab(scorers: analyzer.Scorers) -> None:
    """Upload a CSV: aspect pain points, monthly trend with drop flags, and example quotes."""
    upload = st.file_uploader("Upload a CSV of reviews", type=["csv"])
    if upload is None:
        return
    df = pd.read_csv(upload)
    cols = list(df.columns)
    text_guess = analyzer.guess_column(cols, analyzer.TEXT_GUESSES)
    rating_guess = analyzer.guess_column(cols, analyzer.RATING_GUESSES)
    date_guess = analyzer.guess_column(cols, analyzer.DATE_GUESSES)
    c1, c2, c3 = st.columns(3)
    text_col = c1.selectbox("Review text column", cols, index=cols.index(text_guess) if text_guess else 0)
    rating_col = c2.selectbox("Star rating column (optional)", ["(none)"] + cols,
                              index=cols.index(rating_guess) + 1 if rating_guess else 0)
    date_col = c3.selectbox("Date column (optional)", ["(none)"] + cols, index=cols.index(date_guess) + 1 if date_guess else 0)
    if not st.button("Build dashboard", type="primary"):
        return
    with st.spinner("Scoring reviews..."):
        try:
            res = analyzer.analyze_frame(df, text_col, scorers, app_settings(), CFG.aspects, CFG.labels,
                                         None if rating_col == "(none)" else rating_col,
                                         None if date_col == "(none)" else date_col)
        except ValueError as exc:
            st.error(str(exc))
            return
    m = res.metrics
    if m["sampled"]:
        st.caption(f"Scored a seeded sample of {m['reviews_analysed']:,} of {m['reviews_in_file']:,} reviews.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reviews", f"{m['reviews_analysed']:,}")
    k2.metric("Positive", f"{m['positive_share']:.1%}")
    k3.metric("Negative", f"{m['negative_share']:.1%}")
    k4.metric("Model vs stars (macro-F1)", m.get("model_vs_stars_macro_f1") or "n/a")

    st.subheader("Aspect pain points")
    aspects = res.aspects[["aspect", "reviews", "mention_rate", "neg_review_rate", "pain_ratio", "neg_sentence_share"]]
    st.dataframe(aspects, hide_index=True, use_container_width=True)
    chart = res.aspects.set_index("aspect")["neg_sentence_share"].dropna()
    if not chart.empty:
        st.bar_chart(chart)

    if res.monthly is not None:
        st.subheader("Monthly positive share")
        st.line_chart(res.monthly["share_positive"])
        flagged = res.flags[res.flags["flagged"]] if res.flags is not None else pd.DataFrame()
        if len(flagged):
            st.warning(f"{len(flagged)} month(s) fell more than {CFG.aspects.drop_z:g} sd below the trailing average.")
            st.dataframe(flagged, use_container_width=True)
    elif date_col != "(none)":
        st.caption("Not enough months with enough reviews for a trend.")

    st.subheader("Example complaints")
    st.dataframe(res.examples, hide_index=True, use_container_width=True)
    st.download_button("Download aspect table (CSV)", res.aspects.to_csv(index=False), "aspect_summary.csv", "text/csv")


def main() -> None:
    """App entry point."""
    st.set_page_config(page_title="Review Intelligence", layout="wide")
    st.title("Customer Review Intelligence")
    try:
        scorers = get_scorers()
    except (FileNotFoundError, ImportError, RuntimeError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    tab1, tab2 = st.tabs(["Single review", "Upload CSV dashboard"])
    with tab1:
        single_review_tab(scorers)
    with tab2:
        dashboard_tab(scorers)


main()

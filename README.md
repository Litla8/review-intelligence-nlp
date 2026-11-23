# Customer Review Intelligence (NLP)

Sentiment, topics and aspect-level insights from customer reviews, from a fast TF-IDF baseline to a fine-tuned
DistilBERT, with a Streamlit app. Dataset: Kaggle *Amazon Fine Food Reviews*.

## Status

- [x] Step 1: Problem scope (goals, labels, metrics, config)
- [x] Step 2: Data collection and cleaning (duplicates, empty and non-English reviews, sentiment labels, class balance)
- [x] Step 3: Text preprocessing (HTML, URLs, contractions, emoticons, spaCy lemmas, negations kept)
- [x] Step 4: Exploratory text analysis (length, words, phrases, word clouds, distinctive words, sentiment trend)
- [ ] Step 5: Classical ML (TF-IDF + LR / NB / SVM)
- [ ] Step 6: Transformers (DistilBERT)
- [ ] Step 7: Topic modelling (LDA, BERTopic)
- [ ] Step 8: Aspect-based insights
- [ ] Step 9: Summarization and Streamlit app

## Scope (Step 1)

Three goals: (a) sentiment, (b) topic discovery, (c) aspect insights. Targets: binary macro-F1 above 0.88,
topic coherence above 0.50, a clear aspect report. Details in [docs/project_brief.md](docs/project_brief.md).

## Setup

Download `Reviews.csv` from Kaggle ("Amazon Fine Food Reviews") into `data/raw/`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

Then clean and label the reviews:

```powershell
python -m reviews.prepare    # writes data/processed/reviews_clean.parquet and reports/data_quality.json
```

From Step 3 on: `pip install -r requirements-nlp.txt`. Install PyTorch first with the CUDA 12.8 build, which the
RTX 50-series GPUs need: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Text preprocessing (Step 3)

```powershell
pip install spacy
python -m spacy download en_core_web_sm
python -m reviews.preprocess --sample   # 5,000 reviews, to check the setup
python -m reviews.preprocess            # all reviews: data/processed/reviews_text.parquet, reports/text_report.json
```

Two versions of every review are kept: the untouched `Text` for transformers, and `text_clean` for TF-IDF, topic
models and word clouds. `text_clean` is built by removing HTML and URLs, turning `:)` and `:(` into words, expanding
contractions (`don't` becomes `do not`), lower-casing, dropping digits and symbols, lemmatizing with spaCy and
removing stop words. Negations and intensifiers (`not`, `no`, `never`, `but`, `very`, `too` and others, listed in
`SENTIMENT_WORDS`) are kept, because "not good" must not turn into "good".

## Exploratory analysis (Step 4)

```powershell
pip install wordcloud
python -m reviews.explore
```

Writes seven charts to `reports/figures/` (length, top words, word clouds, bigrams, trigrams, distinctive words,
sentiment trend) and the numbers behind them to `reports/eda_summary.json`. Word and phrase charts hide the words in
`DISPLAY_STOPWORDS` (negations, `very`, and generic verbs such as `make` and `use`) so topics stand out; phrases are
only hidden when *every* word is on that list, so "not good" stays. "Distinctive words" ranks words by a log-odds
z-score, so it shows what separates negative from positive reviews rather than what is common to both. Phrases are
counted on at most 150,000 reviews per sentiment (seeded), and the trend skips months with under 200 reviews.

## Cleaning rules (Step 2)

Applied in this order, with the number of rows each rule removes written to `reports/data_quality.json`:

1. **Unusable text:** missing, blank, or fewer than 3 words.
2. **Invalid rating or date:** anything but a whole-number rating from 1 to 5.
3. **Duplicate review:** same user, product, timestamp and text (keeps the first).
4. **Duplicate text:** the same words under another product or user. The same review is often posted once per
   product variant, and copies in both train and test would inflate the test score.
5. **Non-English:** a dependency-free check (mostly ASCII, and enough common English words in longer texts).

Review text is never changed, so transformer models in Step 6 see the raw text. Ratings become labels
(1-2 negative, 3 neutral, 4-5 positive). Impossible helpfulness votes (more helpful votes than total votes) are set
to missing, not dropped, because the text is still fine.

## Structure

```
reviews/   config.py (all settings), labels.py (stars to sentiment, class balance),
           data.py (load and clean), prepare.py (Step 2 pipeline),
           text.py (text normalisation, lemmas), preprocess.py (Step 3 pipeline),
           eda.py (counts, phrases, trend), plots.py (charts), explore.py (Step 4 pipeline),
           metrics.py (macro-F1, report, inference time, target checks), logging_utils.py
tests/     pytest suite
docs/      project_brief.md
data/      raw and processed data (git-ignored)
```

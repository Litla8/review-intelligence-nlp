# Customer Review Intelligence (NLP)

Sentiment, topics and aspect-level insights from customer reviews, from a fast TF-IDF baseline to a fine-tuned
DistilBERT, with a Streamlit app. Dataset: Kaggle *Amazon Fine Food Reviews*.

## Status

- [x] Step 1: Problem scope (goals, labels, metrics, config)
- [x] Step 2: Data collection and cleaning (duplicates, empty and non-English reviews, sentiment labels, class balance)
- [ ] Step 3: Text preprocessing
- [ ] Step 4: Exploratory text analysis
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
           metrics.py (macro-F1, report, inference time, target checks), logging_utils.py
tests/     pytest suite
docs/      project_brief.md
data/      raw and processed data (git-ignored)
```

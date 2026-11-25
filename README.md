# Customer Review Intelligence (NLP)

Sentiment, topics and aspect-level pain points from customer reviews: a TF-IDF baseline, a fine-tuned DistilBERT,
topic models, aspect insights with trend alerts, summaries, and a deployed Streamlit app.
Dataset: Kaggle *Amazon Fine Food Reviews* (about 568,000 reviews).

| | |
|---|---|
| **Live app** | https://review-intelligence-nlp-lalit.streamlit.app/ |
| **Fine-tuned model** | https://huggingface.co/Litla8/review-sentiment-distilbert |
| **Notebooks** | [01 preprocessing](notebooks/01_preprocessing.ipynb), [02 model training](notebooks/02_model_training.ipynb), [03 topics and aspects](notebooks/03_topics_and_aspects.ipynb) |
| **Tests** | `pytest` (170 tests, no GPU or data needed) |

## Results

| Goal | Target | Result |
|---|---|---|
| Binary sentiment (positive vs negative), macro-F1 | above 0.88 | **0.941** with fine-tuned DistilBERT (best TF-IDF model: 0.882) |
| Topic coherence (c_v) | above 0.50 | above 0.50 for all four topic models (LDA and BERTopic, all reviews and negative-only) |
| Aspect insight report | clear report | ten aspects ranked by pain ratio, monthly trend with drop flags, example quotes |
| Executive summary of pain points | written summary | delivered as a document (ranked pain points, flagged months, actions) |

* DistilBERT beats the best TF-IDF model by more than the 0.01 "worth the cost" rule. Inference cost is reported per 1,000 reviews.
* Biggest pain points (negative reviews mention them far more than positive ones do): **listing** accuracy (pain ratio 3.4),
  **safety** (3.3) and **service** (2.8), followed by packaging (1.7) and freshness (1.3). Taste is the most discussed aspect
  but not the most painful (ratio 1.0).
* The monthly trend flagged 37 aspect-level drops; the worst months for several aspects at once were 2009-08, 2010-09 and 2009-03.
* Caveats: aspects come from a keyword lexicon, so a word like "box" or "value" is not always about that aspect; summaries are
  machine-written; coherence uses a self-written c_v (Gensim has no build for Python 3.14), so values can differ slightly from Gensim's.

## Status

- [x] Step 1: Problem scope (goals, labels, metrics, config)
- [x] Step 2: Data collection and cleaning (duplicates, empty and non-English reviews, sentiment labels, class balance)
- [x] Step 3: Text preprocessing (HTML, URLs, contractions, emoticons, spaCy lemmas, negations kept)
- [x] Step 4: Exploratory text analysis (length, words, phrases, word clouds, distinctive words, sentiment trend)
- [x] Step 5: Classical ML (bag-of-words / TF-IDF with Naive Bayes, Logistic Regression, Linear SVM)
- [x] Step 6: Transformers (DistilBERT fine-tuned with the Hugging Face Trainer)
- [x] Step 7: Topic modelling (LDA, BERTopic, c_v coherence)
- [x] Step 8: Aspect-based insights (pain points per aspect, monthly trend with drop flags)
- [x] Step 9: Summarization and Streamlit app
- [x] Deliverables: notebooks, fine-tuned model (Hugging Face), deployed dashboard, executive summary

## Scope (Step 1)

Three goals: (a) sentiment, (b) topic discovery, (c) aspect insights. Details in [docs/project_brief.md](docs/project_brief.md).

## Quick start

Download `Reviews.csv` from Kaggle ("Amazon Fine Food Reviews") into `data/raw/`. Tested on Windows 11 with an NVIDIA RTX 5070.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series need the CUDA 12.8 build
pip install -r requirements-nlp.txt
python -m spacy download en_core_web_sm
pytest
```

Run the pipeline in order (every step has a `--sample` flag for a quick check, and sample runs write `_sample` files):

```powershell
python -m reviews.prepare            # 2. clean and label        -> reports/data_quality.json
python -m reviews.preprocess         # 3. text preprocessing     -> data/processed/reviews_text.parquet
python -m reviews.explore            # 4. exploratory analysis   -> reports/figures/
python -m reviews.train_classical    # 5. TF-IDF models          -> models/classical_best.joblib
python -m reviews.train_transformer  # 6. DistilBERT (GPU)       -> models/distilbert_sentiment/
python -m reviews.run_topics         # 7. topics (also: --scope negative)
python -m reviews.insights           # 8. aspects and drop flags (GPU)
python -m reviews.summarize          # 9. pain-point summaries   -> reports/pain_point_summaries.md
streamlit run app/streamlit_app.py   # the app, locally
```

The notebooks in `notebooks/` run these steps (flags at the top, off by default) and show the results; open them in VS Code
with the `.venv` kernel and **Run All**.

## The app

* **Single review:** sentiment, probability, the sentences that mention each aspect with their own sentiment, and a summary for long reviews.
* **CSV upload:** pick the text, star and date columns; get positive/negative shares, aspect pain points, monthly trend with
  drop flags, example complaints and a CSV download. With star ratings it also reports how well the model agrees with them.

The deployed app runs on the free Streamlit Community Cloud CPU tier, so it scores at most 2,000 reviews per upload and
has summaries turned off. Run it locally with a GPU for larger files and summaries.

## Deploying (Streamlit Community Cloud + Hugging Face)

1. Upload the fine-tuned model once: `python -m reviews.publish_model --repo YOUR_HF_USERNAME/review-sentiment-distilbert`
   (needs a Hugging Face write token in `HF_TOKEN`).
2. Push the repo to GitHub, then create the app at share.streamlit.io with main file `app/streamlit_app.py` and Python 3.12.
   Dependencies come from `app/requirements.txt` (CPU-only PyTorch).
3. In the app's Secrets add `HF_MODEL_REPO = "YOUR_HF_USERNAME/review-sentiment-distilbert"`.
   Optional: `ENABLE_SUMMARY = "1"` and `APP_MAX_ROWS = "2000"`.

# Step-by-step details

### Text preprocessing (Step 3)

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

### Exploratory analysis (Step 4)

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

### Classical models (Step 5)

```powershell
python -m reviews.train_classical --sample   # 40,000 reviews, a few minutes
python -m reviews.train_classical            # all reviews
```

Positive vs negative only (neutral is dropped). Features come from `text_clean`: unigrams and bigrams, at least 5
reviews per term, at most 200,000 terms. Four models are compared: bag-of-words + Naive Bayes, TF-IDF + Naive Bayes,
TF-IDF + Logistic Regression and TF-IDF + Linear SVM, plus an "always predict the common class" baseline. Each is tuned
on the validation set by macro-F1 and scored once on the test set; imbalance is handled with balanced class weights
(or a uniform prior for Naive Bayes). The vectorisers are fitted on training reviews only. The best model, chosen on
validation, is checked again on a chronological split (train on the oldest reviews, test on the newest) because
sentiment drifts over the years.

Writes `models/classical_best.joblib`, `reports/model_comparison.csv`, `reports/classical_results.json` (including the
terms that push hardest towards each class), `reports/classical_errors.csv` (the mistakes the model was most sure
about) and two charts in `reports/figures/`. Sample runs add `_sample` to every file name.

### Transformer (Step 6)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs the CUDA 12.8 build
pip install transformers accelerate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m reviews.train_transformer --sample    # 30,000 reviews, a few minutes
python -m reviews.train_transformer             # all reviews
```

Fine-tunes `distilbert-base-uncased` on the raw review text (256 tokens, 2 epochs, bfloat16) using exactly the same
train / validation / test split as Step 5, so the scores are directly comparable. The probability cut-off for
"positive" is tuned on the validation set, because 84% of reviews are positive. Reports test accuracy and macro-F1
(default and tuned), training time, prediction time per 1,000 reviews, whether the 0.88 target is met, and whether
the gain over the best TF-IDF model is at least 0.01 (the "worth the cost" rule). It also scores the transformer and
the best TF-IDF model on the oldest and newest third of the test set, and saves the most confident mistakes.

Writes `models/distilbert_sentiment/` (model, tokenizer, `inference_config.json`), `reports/transformer_results.json`,
`reports/model_comparison_all.csv`, `reports/transformer_errors.csv` and two charts. Add `--batch-size 16` if the GPU
runs out of memory.

### Topic modelling (Step 7)

```powershell
pip install bertopic sentence-transformers
python -m reviews.run_topics --sample            # 15,000 reviews, a few minutes
python -m reviews.run_topics                     # 100,000 reviews, all sentiments
python -m reviews.run_topics --scope negative    # negative reviews only: the pain-point view
```

Two models on a seeded sample of reviews, with negations and filler verbs removed (`DISPLAY_STOPWORDS`):
**LDA** (scikit-learn) tries 6 to 20 topics on a smaller sample, keeps the number with the best c_v coherence, and refits on
the full sample; **BERTopic** embeds the raw text with `all-MiniLM-L6-v2` on the GPU, clusters with UMAP and HDBSCAN,
and names topics from the cleaned words. Both are scored with c_v coherence (target above 0.5) and topic diversity. Gensim has no installable build for
Python 3.14, so `reviews/coherence.py` re-implements Gensim's c_v (110-token boolean windows, NPMI, cosine); the numbers
should be close to Gensim's but may differ slightly.
Every topic gets its share of reviews, mean stars, negative share and "lift" (how much more often it appears in negative
reviews than average), which feeds the pain-point report in Step 8. Writes `reports/topic_summary.json`,
`reports/topics_lda.csv`, `reports/topics_bertopic.csv`, `data/processed/topic_assignments.parquet`, the models and
three charts. File names get `_negative` or `_sample` as suffixes.

### Aspect insights and trend flags (Step 8)

```powershell
python -m reviews.insights --sample    # 20,000 reviews, a couple of minutes
python -m reviews.insights             # all reviews (needs the Step 6 model)
```

Reviews are split into sentences. A sentence that contains an aspect keyword (taste, price, shipping, packaging,
freshness, quality, health, service, plus `listing` and `safety`, added because Steps 4 and 7 pointed at them) is scored
by the fine-tuned DistilBERT; sentences it is unsure about (probability between 0.25 and 0.75) count as neutral and are
left out of the shares. For each aspect the report gives the share of negative and positive reviews that mention it,
the ratio between them, and the negative share of confident sentences. A monthly trend flags any month whose positive
share, mean stars or aspect negativity is more than 2 standard deviations worse than the average of the previous 12
months. Keyword matching is a simple lexicon, so a word like "box" or "value" is not always about the aspect.

Writes `reports/aspect_summary.csv`, `aspect_trend_yearly.csv`, `monthly_trend.csv`, `aspect_examples.csv` (the most
negative sentences per aspect, for quotes), `insights_summary.json` and three charts. Sample runs add `_sample`.

### Cleaning rules (Step 2)

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

### Summaries and the Streamlit app (Step 9)

`python -m reviews.summarize` picks the most representative negative reviews per aspect (closest to the group's average
TF-IDF vector) and summarises them with `sshleifer/distilbart-cnn-12-6`; long inputs are split into chunks, summarised,
and the summaries summarised again. The text is machine-written: check it against the example review Ids before quoting.
`reviews/analyzer.py` holds the app's logic, `reviews/predict.py` the shared model loader, and
`reviews/publish_model.py` uploads the model to the Hub.

## Structure

```
reviews/    config.py (all settings), labels.py, data.py + prepare.py (Step 2), text.py + preprocess.py (Step 3),
            eda.py + plots.py + explore.py (Step 4), classical.py + train_classical.py (Step 5),
            transformer.py + train_transformer.py (Step 6), topics.py + coherence.py + run_topics.py (Step 7),
            aspects.py + insights.py (Step 8), summarize.py (Step 9), predict.py, analyzer.py, publish_model.py,
            metrics.py, logging_utils.py
app/        streamlit_app.py, requirements.txt (cloud dependencies)
notebooks/  01_preprocessing, 02_model_training, 03_topics_and_aspects
tests/      pytest suite
docs/       project_brief.md
data/, models/, reports/   generated files (git-ignored)
```

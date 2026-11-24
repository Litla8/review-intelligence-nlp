# Customer Review Intelligence (NLP)

Sentiment, topics and aspect-level insights from customer reviews, from a fast TF-IDF baseline to a fine-tuned
DistilBERT, with a Streamlit app. Dataset: Kaggle *Amazon Fine Food Reviews*.

## Status

- [x] Step 1: Problem scope (goals, labels, metrics, config)
- [x] Step 2: Data collection and cleaning (duplicates, empty and non-English reviews, sentiment labels, class balance)
- [x] Step 3: Text preprocessing (HTML, URLs, contractions, emoticons, spaCy lemmas, negations kept)
- [x] Step 4: Exploratory text analysis (length, words, phrases, word clouds, distinctive words, sentiment trend)
- [x] Step 5: Classical ML (bag-of-words / TF-IDF with Naive Bayes, Logistic Regression, Linear SVM)
- [x] Step 6: Transformers (DistilBERT fine-tuned with the Hugging Face Trainer)
- [x] Step 7: Topic modelling (LDA, BERTopic, c_v coherence)
- [x] Step 8: Aspect-based insights (pain points per aspect, monthly trend with drop flags)
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

## Classical models (Step 5)

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

## Transformer (Step 6)

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

## Topic modelling (Step 7)

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

## Aspect insights and trend flags (Step 8)

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
           classical.py (features, models, splits), train_classical.py (Step 5 pipeline),
           transformer.py (thresholds, drift slices, helpers), train_transformer.py (Step 6 pipeline),
           topics.py (LDA, BERTopic, coherence), run_topics.py (Step 7 pipeline),
           aspects.py (aspect tags, sentence sentiment, drop flags), insights.py (Step 8 pipeline),
           metrics.py (macro-F1, report, inference time, target checks), logging_utils.py
tests/     pytest suite
docs/      project_brief.md
data/      raw and processed data (git-ignored)
```

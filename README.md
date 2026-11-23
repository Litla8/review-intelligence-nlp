# Customer Review Intelligence (NLP)

Sentiment, topics and aspect-level insights from customer reviews, from a fast TF-IDF baseline to a fine-tuned
DistilBERT, with a Streamlit app. Dataset: Kaggle *Amazon Fine Food Reviews*.

## Status

- [x] Step 1: Problem scope (goals, labels, metrics, config)
- [ ] Step 2: Data collection and cleaning
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

From Step 3 on: `pip install -r requirements-nlp.txt`. Install PyTorch first with the CUDA 12.8 build, which the
RTX 50-series GPUs need: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Structure

```
reviews/   config.py (all settings), labels.py (stars to sentiment, class balance),
           metrics.py (macro-F1, report, inference time, target checks), logging_utils.py
tests/     pytest suite
docs/      project_brief.md
data/      raw and processed data (git-ignored)
```

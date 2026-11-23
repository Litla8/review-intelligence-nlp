# Project brief: Customer Review Intelligence

**Business problem.** A food retailer receives hundreds of thousands of reviews and cannot read them. It needs
automatic sentiment, the themes customers talk about, and a short list of what they love and hate.

**Data.** Kaggle *Amazon Fine Food Reviews* (`Reviews.csv`, about 568,000 reviews, 1-5 stars, unix timestamps,
product and user ids, 1999-2012).

## Goals
| # | Goal | Output | Success metric |
|---|------|--------|----------------|
| a | Sentiment (negative / neutral / positive) | TF-IDF baseline and fine-tuned DistilBERT | Binary macro-F1 above 0.88; three-class macro-F1 reported |
| b | Topic discovery | LDA (Gensim) and BERTopic, topics labelled by hand | c_v coherence above 0.50 |
| c | Aspect insights | Sentiment per aspect (taste, price, shipping, packaging, freshness, quality, health, service), monthly trend, drop alerts | Clear aspect report |

## Decisions made now
- **Labels:** 1-2 stars negative, 3 neutral, 4-5 positive. The binary task drops neutral.
- **Primary metric:** macro-F1, because the data is heavily positive and accuracy would hide poor negative-class results.
- **Split:** stratified 70/15/15 with `random_state=42`; text for the transformer stays raw, classical models use cleaned text.
- **Honest comparison:** the transformer must beat TF-IDF by at least 0.01 macro-F1 to justify its cost; inference time is reported per 1,000 reviews.
- **Leakage guard:** duplicate reviews are removed before splitting, so the same text never lands in train and test.

## Deliverables
Preprocessing and training notebooks, fine-tuned model, topic and aspect dashboard (Streamlit), executive summary of pain points.

# Data — Fake News Detector

## Files

- `sample_articles.csv` — 200 article metadata records with analysis results
- `evaluation_summary.json` — aggregate evaluation statistics

## Source

Evaluation corpus assembled from:
- PolitiFact and Snopes fact-checked articles (verified labels)
- Known satirical sources (The Onion, Babylon Bee) — labelled as test edge cases
- Identified misinformation from MediaBiasFactCheck

## Note on training data

The TF-IDF model was trained on a synthetic corpus mirroring patterns from
published fake news datasets (FakeNewsNet, LIAR). See `model/train.py` for details.

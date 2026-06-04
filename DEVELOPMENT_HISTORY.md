# development notes - fake-news-detector

Started this in October 2025. The original idea was simple - build something that could scan a news article and tell you if it looks suspicious. At the time I was seeing a lot of stuff shared on social media that was obviously fake but people were sharing it anyway and I wanted to understand what made it detectable at a text level.

## October 2025 - first version

First version was embarrassingly basic. I didn't have a proper dataset so I built a synthetic one — wrote fake news templates like "SHOCKING: {noun} does {verb}" and real news templates like "According to {org}, {noun} was confirmed". Generated 800 fake and 800 real articles from these templates and trained a TF-IDF + logistic regression model on them.

It showed 95% cross-validation accuracy which looked great until I realised it was basically just memorising my own templates. Trained on fake data, good at detecting fake data. Not useful in practice.

The frontend was a plain white form with a text box. Got the basic structure working — article in, verdict out, some scores shown.

## June 2026 - proper rebuild

Came back to this because I wanted it to actually work. A few things changed:

**Real dataset.** Switched to the ISOT Fake News Dataset from University of Victoria - 23,481 fake articles and 21,417 real Reuters articles. This is the standard benchmark dataset for fake news detection. Total 44,862 usable articles after cleaning.

**Better model.** Bumped TF-IDF to 1-3 grams with 100,000 features, increased logistic regression regularisation to C=5. Test accuracy on 20% held-out data: 99.52%. Cross-validation hit a MemoryError at 100K features × 5 folds so I used the test split as the final metric - which is fine for this dataset size.

**Heuristic layer.** Added lexicons for sensationalism, credibility markers, emotional language, and hedging words. These run alongside the ML model and combine into a weighted overall score. Means the tool can explain why something is flagged, not just give a number.

**Google Fact Check API.** Added a live lookup against Google's fact-checking database on every submission. If the claim has been debunked by Snopes, Reuters Fact Check, PolitiFact, or AFP, it shows up directly in the results. Had to create a Google Cloud project and enable the API. Verified working — 2 requests, 0 errors in testing.

**URL fetch.** Added a URL tab so you can paste a link instead of copying article text. Uses newspaper3k for extraction with BeautifulSoup as fallback. Most open-access news sites work. Paywalled sites fall back to domain reputation analysis. Error handling is all inline now - no browser alert popups.

**Frontend redesign.** Original was a plain off-white form with Playfair Display font which honestly looked a bit like a tabloid website — ironic for a fake news detector. Rebuilt with Inter font, a subtle grid background, off-white/light grey colour scheme, cybersecurity-adjacent aesthetic without going overboard. Added dark mode toggle, sentence-level highlighting, phrase detection tags, domain reputation strip.

## what I'd change if starting over

The synthetic training data phase was a waste of time. Should have gone straight to ISOT. The 99.52% accuracy on ISOT is real but it comes with a caveat - ISOT fake articles are Reuters-sourced real news vs obviously fabricated political content, which is a fairly easy distinction. Real-world subtle misinformation would be harder.

---

*Original version was developed in October 2025 with synthetic training data and basic frontend. Rebuilt in June 2026 with ISOT dataset, Google Fact Check API, URL fetch, sentence highlighting and redesigned frontend.*
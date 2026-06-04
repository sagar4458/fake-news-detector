# TruthLens - Fake News Detector

![Accuracy](https://img.shields.io/badge/Accuracy-99.5%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-ISOT%2044K-orange)
![API](https://img.shields.io/badge/Fact--Check-Google%20API-red)
![Status](https://img.shields.io/badge/Status-Active-success)

> Paste any news article or URL. Get a misinformation score, sentence-level analysis, and live fact-check results from Google's database - in under 2 seconds.

Originally built in October 2025. Rebuilt from scratch in June 2026 with a real dataset, Google Fact Check API integration, URL fetch, and a completely redesigned frontend.

---

## screenshots

### original version - October 2025
| Dashboard | Results |
|---|---|
| ![Old Dashboard](screenshots/1_old_frontend.png) | ![Old Results](screenshots/2_old_frontend_results.png) |

### rebuilt version - June 2026
| New Dashboard | Dark Mode |
|---|---|
| ![New Dashboard](screenshots/3_new_updated_dashboard.png) | ![Dark Mode](screenshots/4_new_updated_dashboard_dark_mode.png) |

| Analysis Result | URL Fetch Result |
|---|---|
| ![Analysis 1](screenshots/5_new_dashboard_analysis_report.png) | ![Analysis 2](screenshots/6_new_dashboard_analysis_report_2.png) |

| URL Error Handling | Google Fact Check API |
|---|---|
| ![Error Handling](screenshots/7_new_dashboard_analysis_url_error_report.png) | ![API Dashboard](screenshots/factcheck_api.png) |

### google api metrics - june 2026
| Traffic | Errors | Median Latency |
|---|---|---|
| ![Traffic](screenshots/10_Traffic.png) | ![Errors](screenshots/9_Errors.png) | ![Latency](screenshots/8_Median_latency.png) |

**2 requests · 0 errors · 0.4ms median latency**

---

## what it does

- ✅ ML model trained on 44,862 real articles - 99.52% test accuracy
- ✅ Sentence-level highlighting - suspicious sentences flagged amber, credible ones green
- ✅ Live Google Fact Check API lookup on every submission
- ✅ URL fetch mode - paste a link, auto-extracts article content
- ✅ Domain reputation check - flags known unreliable sources
- ✅ Detected phrase tags - shows exactly which phrases triggered the score
- ✅ Readability, lexical diversity, hedging language metrics
- ✅ Dark mode
- ✅ Inline error handling - no browser popups

---

## how it works

Two layers run on every submission:

**Layer 1 - ML model**
TF-IDF vectoriser (1-3 grams, 100K features) converts the article text into a feature vector. Logistic regression (C=5) classifies it as real or fake and returns a probability score.

**Layer 2 - Heuristic analysis**
Lexicons check for sensationalist phrases, credibility markers, emotional language, and hedging words. Flesch-Kincaid readability and lexical diversity are computed. Results combine with the ML score via weighted average.

**Layer 3 - Google Fact Check API**
Every submission fires a live query to Google's fact-checking database. Returns matching debunks from Snopes, Reuters Fact Check, PolitiFact, AFP, and BBC Verify if available.

---

## model

| Version | Training Data | Accuracy |
|---|---|---|
| v1 - Oct 2025 | Synthetic templates (not meaningful) | ~95% |
| v2 - Jun 2026 | ISOT 44K real articles | **99.52% test accuracy** |

Dataset: [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) - University of Victoria
- `Fake.csv` - 23,481 articles from unreliable sources flagged by PolitiFact
- `True.csv` - 21,417 articles from Reuters

---

## api integration - verified live

Google Fact Check Tools API connected and verified working.

![API Dashboard](screenshots/factcheck_api.png)

**2 requests · 0 errors · 524ms median latency** - from testing session, June 2026.

---

## running it

```bash
cd backend
pip install -r requirements.txt
python ../model/train.py
python app.py
```

Open `frontend/index.html` in Chrome. Backend runs on **port 5002**.

Training takes ~5 minutes on the full ISOT dataset. The model file is excluded from the repo - run `train.py` once after cloning.

---

## stack

`Python` `Flask` `scikit-learn` `TF-IDF` `NumPy` `newspaper3k` `BeautifulSoup` `Google Fact Check API` `JavaScript` `HTML/CSS`

---

## limitations

The 99.52% accuracy is on the ISOT dataset which is a fairly clean binary problem - Reuters journalism versus obviously fabricated political content. Real-world subtle misinformation is harder. The heuristic layer helps with edge cases but this is a research tool, not a production fact-checker.

URL fetch works on most open-access sites. Paywalled and JavaScript-rendered pages fall back to domain reputation analysis.

---

*Originally developed: October 2025 - rebuilt and open-sourced: June 2026*

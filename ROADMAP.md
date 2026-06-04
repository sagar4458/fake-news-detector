# roadmap - fake-news-detector

## where it stands

99.52% test accuracy on ISOT dataset. Logistic regression + TF-IDF, 100K features, 1-3 grams. Google Fact Check API connected. URL fetch working for most open-access sites. Sentence-level highlighting, phrase detection, domain reputation, dark mode. Runs locally, no cloud dependency.

## the model ceiling

99.52% on ISOT looks great but ISOT is a fairly clean binary classification problem - Reuters real news versus obviously fabricated political articles. The harder problem is detecting subtle misinformation that reads like legitimate journalism but contains misleading framing, selective facts or unverified claims. The current model would struggle with that.

The next model step is fine-tuning a BERT or DistilBERT on a more diverse dataset. The LIAR dataset from UCSB has 12,836 labelled statements with six-level truthfulness ratings (true, mostly-true, half-true, barely-true, false, pants-on-fire) which is significantly harder than binary fake/real. Getting above 70% on LIAR would be a meaningful result.

## URL fetch improvements

The current newspaper3k approach works on open-access sites but fails on paywalled content and JavaScript-rendered pages. Two upgrades worth trying: Playwright for JavaScript-rendered pages (runs a real browser headless), and an RSS feed fallback for sites that publish full articles in their feeds. Most major news organisations still have RSS.

## sentence-level explanation

Right now the sentence highlighting just flags sentences containing words from the sensationalism or credibility lexicons. A proper explanation would use LIME or SHAP to show which words in the article actually influenced the ML model's decision. "This article is flagged because the phrases 'shocking truth' and 'they don't want you to know' pushed the model toward fake" is much more useful than a raw percentage.

LIME works on text classifiers out of the box. Worth adding as a separate "explain this result" button rather than running it on every submission since it's slower.

## source credibility expansion

The domain reputation list is currently hardcoded - about 15 reliable domains and 8 unreliable ones. A better approach is to pull from the Media Bias Fact Check database which covers thousands of domains with bias ratings and factual reporting scores. They have an unofficial API that works for basic lookups.

Also worth adding: check if the article URL exists on Archive.org. If a site has been regularly archived for years, that's a weak signal of legitimacy. If it was registered last month, that's a red flag.

## batch analysis

A CSV upload mode where you can submit a list of URLs or article texts and get credibility scores back in bulk. Useful for researchers or journalists checking multiple sources at once. The backend already handles individual analyses fast enough that batch processing would just need a queue.

## browser extension

The obvious long-term goal is a Chrome extension that adds a TruthLens score to any news article you're reading without having to copy and paste. Would need the backend hosted somewhere accessible rather than running locally, which means sorting out deployment - probably a small VPS since patient data isn't involved here.

## what I'm not going to do

Build a public API with rate limiting and authentication. The scope of this project is a local research tool and portfolio piece. Turning it into a public service is a different problem with different requirements - content moderation, abuse prevention, legal liability around flagging specific publications. Not something to take on casually.
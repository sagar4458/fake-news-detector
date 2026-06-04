from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, re, os, joblib, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os as _os

app  = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
_pipeline = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.joblib"))

GOOGLE_FACTCHECK_API_KEY = _os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
GOOGLE_FACTCHECK_URL     = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

SENSATIONAL = [
    "shocking", "bombshell", "urgent", "breaking", "exposed", "cover-up",
    "secret", "they don't want you to know", "mainstream media", "wake up",
    "share before deleted", "you won't believe", "doctors hate", "weird trick",
    "100% proven", "baffled", "explosive", "scandalous", "conspiracy",
    "they're hiding", "banned", "censored", "deep state", "globalist",
    "hoax", "plandemic", "false flag", "crisis actor", "new world order"
]
CREDIBLE = [
    "according to", "study published", "researchers found", "data shows",
    "confirmed", "spokesperson", "percent", "investigation", "analysis",
    "evidence suggests", "peer reviewed", "journal", "university", "institute",
    "survey of", "statistics show", "report by", "cited", "source said",
    "officials said", "press release", "court documents", "filing shows"
]
EMOTIONAL = [
    "outrage", "disgusting", "evil", "corrupt", "traitors", "criminals",
    "destroyed", "terrifying", "devastating", "catastrophic", "alarming",
    "disgrace", "horrifying", "shameful", "vile", "wicked", "monstrous",
    "despicable", "atrocious", "unforgivable", "enraging"
]
HEDGING = [
    "apparently", "allegedly", "reportedly", "seems", "suggests", "may",
    "might", "could", "possibly", "perhaps", "unclear", "unverified",
    "sources say", "it is believed", "rumored", "claimed"
]
UNRELIABLE_DOMAINS = [
    "infowars", "naturalnews", "beforeitsnews", "worldnewsdailyreport",
    "empirenews", "thelastlineofdefense", "abcnews.com.co", "theonion",
    "yournewswire", "neonnettle", "thegatewaypundit", "zerohedge",
    "breitbart", "dailystormer", "veteranstoday"
]
RELIABLE_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "theguardian.com", "washingtonpost.com", "npr.org", "pbs.org",
    "economist.com", "ft.com", "bloomberg.com", "wsj.com", "time.com",
    "theatlantic.com", "scientificamerican.com", "nature.com", "science.org",
    "who.int", "cdc.gov", "nih.gov", "gov.uk", "europa.eu"
]

FETCH_HEADERS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    },
    {
        "User-Agent": "Googlebot/2.1 (+http://www.google.com/bot.html)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    {
        "User-Agent": "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)",
        "Accept": "text/html",
    }
]


def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_from_url_slug(url):
    try:
        parsed = urlparse(url)
        path   = parsed.path
        slug   = path.rstrip('/').split('/')[-1]
        slug   = re.sub(r'[-_]', ' ', slug)
        slug   = re.sub(r'\d+', '', slug)
        slug   = re.sub(r'\s+', ' ', slug).strip()
        return slug if len(slug) > 10 else ""
    except Exception:
        return ""


def fetch_url(url):
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "", "", "invalid_url"

    title      = ""
    body       = ""
    fetch_status = "success"

    try:
        from newspaper import Article
        a = Article(url)
        a.download()
        a.parse()
        title = a.title or ""
        body  = a.text  or ""
        if len(body) > 100:
            return title, body[:8000], "success"
    except Exception:
        pass

    for headers in FETCH_HEADERS:
        try:
            resp = requests.get(url, headers=headers, timeout=12, allow_redirects=True)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")

                if not title:
                    og_title = soup.find("meta", property="og:title")
                    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
                    t_tag    = soup.find("title")
                    title    = (og_title and og_title.get("content")) or \
                               (tw_title and tw_title.get("content")) or \
                               (t_tag and t_tag.get_text(strip=True)) or ""

                og_desc  = soup.find("meta", property="og:description")
                tw_desc  = soup.find("meta", attrs={"name": "twitter:description"})
                meta_desc = soup.find("meta", attrs={"name": "description"})
                description = (og_desc  and og_desc.get("content"))  or \
                              (tw_desc  and tw_desc.get("content"))   or \
                              (meta_desc and meta_desc.get("content")) or ""

                for tag in soup(["script","style","nav","footer","header",
                                  "aside","iframe","noscript","form","button"]):
                    tag.decompose()

                article_tag = soup.find("article")
                if article_tag:
                    paragraphs = article_tag.find_all(["p","h1","h2","h3"])
                else:
                    paragraphs = soup.find_all(["p","h1","h2","h3"])

                body = " ".join(p.get_text(strip=True) for p in paragraphs
                                if len(p.get_text(strip=True)) > 30)

                if len(body) < 150 and description:
                    body = description

                if len(body) > 100:
                    return title, body[:8000], "success"

            elif resp.status_code == 403:
                fetch_status = "blocked"
            elif resp.status_code == 404:
                return "", "", "not_found"
            elif resp.status_code == 429:
                fetch_status = "rate_limited"

        except requests.exceptions.ConnectionError:
            return "", "", "connection_error"
        except requests.exceptions.Timeout:
            fetch_status = "timeout"
        except Exception:
            continue

    slug = extract_from_url_slug(url)
    if slug:
        return title or slug, slug, "partial"

    return title, "", fetch_status


def check_domain(url):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        domain = url
    for d in UNRELIABLE_DOMAINS:
        if d in domain:
            return "unreliable", domain
    for d in RELIABLE_DOMAINS:
        if d in domain:
            return "reliable", domain
    return "unknown", domain


def flesch_score(text):
    sentences = max(len(re.split(r'[.!?]+', text)), 1)
    words     = text.split()
    if not words:
        return 50.0
    syllables = sum(max(len(re.findall(r'[aeiou]+', w.lower())), 1) for w in words)
    return 206.835 - 1.015*(len(words)/sentences) - 84.6*(syllables/len(words))


def lexical_diversity(text):
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words) * 100, 1)


def highlight_sentences(text):
    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    highlighted = []
    for sent in sentences:
        if not sent.strip():
            continue
        lower = sent.lower()
        sens  = sum(1 for k in SENSATIONAL if k in lower)
        cred  = sum(1 for k in CREDIBLE    if k in lower)
        tag   = "suspicious" if sens > 0 else "credible" if cred > 0 else "neutral"
        highlighted.append({"text": sent.strip(), "tag": tag})
    return highlighted


def factcheck_lookup(query):
    try:
        q      = " ".join(query.split()[:10])
        params = {"key": GOOGLE_FACTCHECK_API_KEY, "query": q, "languageCode": "en"}
        resp   = requests.get(GOOGLE_FACTCHECK_URL, params=params, timeout=8)
        data   = resp.json()
        results = []
        for claim in data.get("claims", [])[:3]:
            review = claim.get("claimReview", [{}])[0]
            results.append({
                "claim":     claim.get("text", ""),
                "rating":    review.get("textualRating", ""),
                "publisher": review.get("publisher", {}).get("name", ""),
                "url":       review.get("url", "")
            })
        return results
    except Exception:
        return []


def analyse(title, text, url=""):
    combined      = ((title + " ") if title else "") + text
    cleaned       = clean_text(combined)
    lower         = combined.lower()
    words         = lower.split()
    n             = max(len(words), 1)

    sens_score    = sum(1 for k in SENSATIONAL if k in lower) / len(SENSATIONAL)
    cred_score    = sum(1 for k in CREDIBLE    if k in lower) / len(CREDIBLE)
    emotion_score = sum(1 for w in words if w in EMOTIONAL)   / n
    hedge_score   = sum(1 for k in HEDGING     if k in lower) / len(HEDGING)
    caps_ratio    = sum(1 for c in combined if c.isupper())   / max(len(combined), 1)
    exclaim_count = combined.count("!")
    fk            = flesch_score(combined)
    lex_div       = lexical_diversity(text)
    ml_prob_fake  = float(_pipeline.predict_proba([cleaned])[0][1])

    domain_status = "unknown"
    domain_name   = ""
    if url:
        domain_status, domain_name = check_domain(url)

    domain_adj = 0.15 if domain_status == "unreliable" else \
                -0.08 if domain_status == "reliable"   else 0.0

    overall = min(1.0, max(0.0,
        ml_prob_fake         * 0.52 +
        sens_score           * 0.16 +
        (1 - cred_score)     * 0.12 +
        emotion_score        * 0.10 +
        min(caps_ratio*5, 1) * 0.05 +
        domain_adj           * 0.05
    ))

    red_flags   = []
    pos_signals = []

    if sens_score > 0.08:             red_flags.append("Sensationalist language detected")
    if caps_ratio > 0.15:             red_flags.append("Excessive use of capital letters")
    if exclaim_count > 3:             red_flags.append(f"Heavy use of exclamation marks ({exclaim_count})")
    if emotion_score > 0.04:          red_flags.append("High emotional loading")
    if fk < 35:                       red_flags.append("Oversimplified language")
    if cred_score < 0.03:             red_flags.append("No credible attribution found")
    if domain_status == "unreliable": red_flags.append(f"Domain flagged as unreliable: {domain_name}")

    if cred_score > 0.10:             pos_signals.append("Multiple credible attribution phrases")
    if caps_ratio < 0.05:             pos_signals.append("Normal capitalisation")
    if exclaim_count == 0:            pos_signals.append("No exclamation marks")
    if fk > 50:                       pos_signals.append("Good readability level")
    if sens_score == 0:               pos_signals.append("No sensationalist language")
    if hedge_score > 0.05:            pos_signals.append("Appropriate use of hedging language")
    if domain_status == "reliable":   pos_signals.append(f"Known reliable source: {domain_name}")

    verdict = "Likely Misinformation"      if overall > 0.65 else \
              "Suspicious — verify independently" if overall > 0.42 else \
              "Likely Credible"

    return {
        "success":             True,
        "verdict":             verdict,
        "overall_score":       round(overall * 100, 1),
        "ml_fake_prob":        round(ml_prob_fake * 100, 1),
        "sensationalism":      round(sens_score * 100, 1),
        "credibility":         round(cred_score * 100, 1),
        "emotional_loading":   round(emotion_score * 100, 1),
        "hedging_score":       round(hedge_score * 100, 1),
        "readability":         round(fk, 1),
        "lexical_diversity":   lex_div,
        "caps_ratio":          round(caps_ratio * 100, 1),
        "red_flags":           red_flags,
        "positive_signals":    pos_signals,
        "sensational_phrases": [k for k in SENSATIONAL if k in lower][:5],
        "credible_phrases":    [k for k in CREDIBLE    if k in lower][:5],
        "highlighted":         highlight_sentences(text) if text else [],
        "factchecks":          factcheck_lookup(title if title else text[:100]),
        "domain_status":       domain_status,
        "domain_name":         domain_name,
    }


STATUS_MESSAGES = {
    "blocked":          "This site blocks automated access. The domain has still been analysed for credibility.",
    "not_found":        "The URL returned a 404 — page does not exist. Check the URL and try again.",
    "rate_limited":     "This site is rate-limiting requests. Try again in a few minutes or paste the article text manually.",
    "timeout":          "The request timed out. The site may be slow or unavailable.",
    "connection_error": "Could not connect to that URL. Check your internet connection or the URL.",
    "invalid_url":      "That doesn't look like a valid URL. Make sure it starts with https://",
    "partial":          "Could only extract partial content from the URL — analysing what was available.",
}


@app.route("/api/analyse", methods=["POST"])
def analyse_route():
    data  = request.json or {}
    url   = data.get("url", "").strip()
    text  = data.get("text", "").strip()
    title = data.get("title", "").strip()
    fetch_info = None

    if url:
        fetched_title, fetched_text, status = fetch_url(url)

        if status == "not_found":
            return jsonify({
                "success":     False,
                "error":       STATUS_MESSAGES["not_found"],
                "error_type":  "not_found",
                "url":         url
            }), 400

        if status == "invalid_url":
            return jsonify({
                "success":     False,
                "error":       STATUS_MESSAGES["invalid_url"],
                "error_type":  "invalid_url",
                "url":         url
            }), 400

        if status == "connection_error":
            return jsonify({
                "success":     False,
                "error":       STATUS_MESSAGES["connection_error"],
                "error_type":  "connection_error",
                "url":         url
            }), 400

        title = title or fetched_title
        text  = fetched_text

        if status in ("blocked", "timeout", "rate_limited"):
            fetch_info = STATUS_MESSAGES[status]

        if status == "partial":
            fetch_info = STATUS_MESSAGES["partial"]

        if not text and not title:
            domain_status, domain_name = check_domain(url)
            return jsonify({
                "success":      False,
                "error":        f"Could not extract article content from this URL. The domain ({domain_name}) is classified as {domain_status}. To analyse the article content, copy and paste the text using the TEXT tab.",
                "error_type":   status,
                "domain_status": domain_status,
                "domain_name":  domain_name,
                "url":          url
            }), 400

        if not text and title:
            text = title

    if len(text) < 30:
        return jsonify({
            "success":    False,
            "error":      "Text too short — enter at least 30 characters.",
            "error_type": "too_short"
        }), 400

    result = analyse(title, text, url)
    if fetch_info:
        result["fetch_warning"] = fetch_info
    return jsonify(result)


@app.route("/api/model_info", methods=["GET"])
def model_info():
    import json as j
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return jsonify(j.load(f))


if __name__ == "__main__":
    app.run(debug=True, port=5002)

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests


# -------- CONFIG --------
RSS_FEEDS = {
    "Technical Analysis": "https://www.investing.com/rss/bonds_Technical.rss",
    "Fundamental Analysis": "https://www.investing.com/rss/bonds_Fundamental.rss",
    "Opinion": "https://www.investing.com/rss/bonds_Opinion.rss",
    "Strategy": "https://www.investing.com/rss/bonds_Strategy.rss",
    "Government Bonds Analysis": "https://www.investing.com/rss/bonds_Government.rss",
    "Corporate Bonds Analysis": "https://www.investing.com/rss/bonds_Corporate.rss",
}


# -------- INIT --------
analyzer = SentimentIntensityAnalyzer()
all_entries = []

# -------- PARSE FEEDS --------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

for source, url in RSS_FEEDS.items():
    try:
        response = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(response.content)
        print(f"[{source}] Found {len(feed.entries)} entries.")
    except Exception as e:
        print(f"[{source}] Error fetching feed: {e}")
        continue

    for entry in feed.entries:
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        published = entry.get('published', '') or entry.get('updated', '') or None
        link = entry.get('link', '')

        if not title:
            continue

        full_text = f"{title} {summary}"
        sentiment_score = analyzer.polarity_scores(full_text)['compound']

        all_entries.append({
            "source": source,
            "title": title,
            "summary": summary,
            "published": published,
            "link": link,
            "sentiment_score": sentiment_score
        })


# -------- CHECK AND SAVE --------
if not all_entries:
    print("⚠️ No RSS entries were collected. Check your feeds or filters.")
    exit()

df = pd.DataFrame(all_entries)

df["published"] = pd.to_datetime(df.get("published", pd.NaT), errors='coerce')

df["sentiment_label"] = df["sentiment_score"].apply(
    lambda x: "positive" if x >= 0.05 else "negative" if x <= -0.05 else "neutral"
)

df.to_csv("rss/csvs/rss_sentiment_results.csv", index=False)
print("✅ Saved to rss_sentiment_results.csv")
print(df.head())

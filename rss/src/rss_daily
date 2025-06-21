import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import os

# --- CONFIG ---
CSV_PATH = "rss/csvs/rss_sentiment_results.csv"
RSS_FEEDS = {
    "Technical Analysis": "https://www.investing.com/rss/bonds_Technical.rss",
    "Fundamental Analysis": "https://www.investing.com/rss/bonds_Fundamental.rss",
    "Opinion": "https://www.investing.com/rss/bonds_Opinion.rss",
    "Strategy": "https://www.investing.com/rss/bonds_Strategy.rss",
    "Government Bonds Analysis": "https://www.investing.com/rss/bonds_Government.rss",
    "Corporate Bonds Analysis": "https://www.investing.com/rss/bonds_Corporate.rss",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# --- INIT ---
analyzer = SentimentIntensityAnalyzer()
existing_df = pd.read_csv(CSV_PATH, parse_dates=["published"]) if os.path.exists(CSV_PATH) else pd.DataFrame()
last_date = existing_df["published"].max() if not existing_df.empty else datetime.min

all_new_entries = []

# --- FETCH AND FILTER ---
for source, url in RSS_FEEDS.items():
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        feed = feedparser.parse(response.content)
        print(f"[{source}] Found {len(feed.entries)} entries.")
    except Exception as e:
        print(f"[{source}] Error fetching feed: {e}")
        continue

    for entry in feed.entries:
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        link = entry.get('link', '')
        published_str = entry.get('published', '') or entry.get('updated', '')
        try:
            published = datetime(*entry.published_parsed[:6])
        except Exception:
            published = None

        if not title or not published or published <= last_date:
            continue

        full_text = f"{title} {summary}"
        sentiment_score = analyzer.polarity_scores(full_text)['compound']

        all_new_entries.append({
            "source": source,
            "title": title,
            "summary": summary,
            "published": published,
            "link": link,
            "sentiment_score": sentiment_score,
            "sentiment_label": (
                "positive" if sentiment_score >= 0.05
                else "negative" if sentiment_score <= -0.05
                else "neutral"
            )
        })

# --- SAVE ---
if not all_new_entries:
    print("✅ No new entries since last run.")
else:
    new_df = pd.DataFrame(all_new_entries)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["link"])
    combined_df.to_csv(CSV_PATH, index=False)
    print(f"✅ Appended {len(new_df)} new entries. Total rows: {len(combined_df)}")

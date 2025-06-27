import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import os
import time
from dotenv import load_dotenv
from pandas.errors import EmptyDataError

# --- LOAD ENV ---
load_dotenv()

# --- CONFIG ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER = os.getenv("REDDIT_USER")
REDDIT_PASS = os.getenv("REDDIT_PASS")
USER_AGENT = os.getenv("USER_AGENT")

SUBREDDITS = ['bonds']
POST_LIMIT = 500
CSV_PATH = 'reddit/csvs/reddit_sentiment_results.csv'

# --- INIT ---
analyzer = SentimentIntensityAnalyzer()
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    username=REDDIT_USER,
    password=REDDIT_PASS,
    user_agent=USER_AGENT,
    check_for_async=False
)

# --- LOAD EXISTING CSV ---
if os.path.exists(CSV_PATH):
    try:
        existing_df = pd.read_csv(CSV_PATH, parse_dates=["created"])
        if existing_df.empty or "created" not in existing_df.columns:
            print("🆕 CSV exists but is empty or malformed. Scraping everything.")
            latest_time = datetime.min
        else:
            latest_time = existing_df["created"].max()
            print(f"🔁 Last saved post was at: {latest_time}")
    except EmptyDataError:
        print("🆕 CSV exists but is completely empty. Scraping everything.")
        existing_df = pd.DataFrame()
        latest_time = datetime.min
else:
    existing_df = pd.DataFrame()
    latest_time = datetime.min
    print("🆕 No existing CSV found. Scraping everything.")

# --- FETCH NEW POSTS + COMMENTS ---
all_posts = []

for sub in SUBREDDITS:
    subreddit = reddit.subreddit(sub)
    print(f"\n🔍 Scraping r/{sub}...")

    for post in subreddit.new(limit=POST_LIMIT):
        post_time = datetime.fromtimestamp(post.created_utc)
        if post_time <= latest_time:
            print(f"⏭️ Skipping old post: {post.title[:40]} ({post_time})")
            continue

        print(f"\n📝 Processing post: {post.title[:60]}")

        # --- Analyze post title ---
        title_score = analyzer.polarity_scores(post.title)['compound']
        title_label = (
            "positive" if title_score >= 0.05
            else "negative" if title_score <= -0.05
            else "neutral"
        )

        # --- Analyze comments ---
        try:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()
            print(f"🗨️  Found {len(comments)} raw comments")

            comment_texts = []
            for c in comments:
                if isinstance(c.body, str):
                    if len(c.body.strip()) > 10:
                        comment_texts.append(c.body)
                    else:
                        print(f"⏭️ Skipping short comment: '{c.body}'")

            comment_scores = [analyzer.polarity_scores(c)['compound'] for c in comment_texts]
            avg_comment_score = sum(comment_scores) / len(comment_scores) if comment_scores else 0
            comment_label = (
                "positive" if avg_comment_score >= 0.05
                else "negative" if avg_comment_score <= -0.05
                else "neutral"
            )
        except Exception as e:
            print(f"⚠️ Error fetching comments: {e}")
            comment_scores = []
            avg_comment_score = 0
            comment_label = "neutral"

        sentiment_gap = title_score - avg_comment_score

        all_posts.append({
            "subreddit": sub,
            "title": post.title,
            "created": post_time,
            "url": post.url,
            "sentiment_score": title_score,
            "sentiment_label": title_label,
            "comment_count": len(comment_scores),
            "avg_comment_sentiment": avg_comment_score,
            "comment_sentiment_label": comment_label,
            "sentiment_gap": sentiment_gap
        })

        time.sleep(1)  # Avoid rate limits

# --- SAVE TO CSV ---
if not all_posts:
    print("\n✅ No new posts to add.")
else:
    new_df = pd.DataFrame(all_posts)
    combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["url"])
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Added {len(new_df)} new posts. Total now: {len(combined)}")

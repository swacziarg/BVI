import praw
import pandas as pd
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
POST_LIMIT = 5000
CSV_PATH = 'reddit/csvs/reddit_raw_posts_and_comments.csv'

# --- INIT ---
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
all_entries = []

for sub in SUBREDDITS:
    subreddit = reddit.subreddit(sub)
    print(f"\n🔍 Scraping r/{sub}...")

    for post in subreddit.new(limit=POST_LIMIT):
        post_time = datetime.fromtimestamp(post.created_utc)
        if post_time <= latest_time:
            print(f"⏭️ Skipping old post: {post.title[:40]} ({post_time})")
            continue

        print(f"\n📝 Post: {post.title[:60]}")

        # Add post title
        all_entries.append({
            "type": "title",
            "subreddit": sub,
            "text": post.title,
            "created": post_time,
            "url": post.url
        })

        # --- Get comments ---
        try:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()
            print(f"🗨️  Found {len(comments)} raw comments")

            for c in comments:
                if isinstance(c.body, str) and len(c.body.strip()) > 10:
                    all_entries.append({
                        "type": "comment",
                        "subreddit": sub,
                        "text": c.body.strip(),
                        "created": datetime.fromtimestamp(c.created_utc),
                        "url": post.url
                    })
        except Exception as e:
            print(f"⚠️ Error fetching comments: {e}")

        time.sleep(1)  # Respect Reddit rate limits

# --- SAVE TO CSV ---
if not all_entries:
    print("\n✅ No new posts or comments to add.")
else:
    new_df = pd.DataFrame(all_entries)
    combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["text", "created"])
    combined.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Added {len(new_df)} new entries. Total now: {len(combined)}")

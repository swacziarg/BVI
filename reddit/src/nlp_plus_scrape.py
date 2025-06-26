if __name__ == "__main__":
    import os
    import time
    from datetime import datetime

    import pandas as pd
    import praw
    from dotenv import load_dotenv
    from transformers import pipeline
    from pandas.errors import EmptyDataError
    from tqdm import tqdm

    # --- LOAD ENV ---
    load_dotenv()

    # --- CONFIG ---
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_SECRET = os.getenv("REDDIT_SECRET")
    REDDIT_USER = os.getenv("REDDIT_USER")
    REDDIT_PASS = os.getenv("REDDIT_PASS")
    USER_AGENT = os.getenv("USER_AGENT")

    SUBREDDITS = ['bonds']
    POST_LIMIT = 50000
    CSV_PATH = 'reddit/csvs/reddit_raw_posts_and_comments.csv'
    OUTPUT_LABELED_PATH = 'reddit/csvs/auto_labeled_reddit_comments.csv'

    # --- INIT REDDIT ---
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
            latest_time = existing_df["created"].max() if not existing_df.empty else datetime.min
            print(f"🔁 Last saved post was at: {latest_time}")
        except EmptyDataError:
            existing_df = pd.DataFrame()
            latest_time = datetime.min
            print("🆕 CSV exists but is empty. Scraping everything.")
    else:
        existing_df = pd.DataFrame()
        latest_time = datetime.min
        print("🆕 No existing CSV found. Scraping everything.")

    # --- SCRAPE POSTS + COMMENTS ---
    all_entries = []
    found_old_post = False

    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        print(f"\n🔍 Scraping r/{sub}...")

        for post in subreddit.new(limit=POST_LIMIT):
            post_time = datetime.fromtimestamp(post.created_utc)
            if post_time <= latest_time:
                print(f"⏹️ Hit old post: {post.title[:40]} ({post_time}) — stopping")
                found_old_post = True
                break

            print(f"\n📝 Post: {post.title[:60]}")

            all_entries.append({
                "type": "title",
                "subreddit": sub,
                "text": post.title,
                "created": post_time,
                "url": post.url
            })

            try:
                post.comments.replace_more(limit=0)
                comments = post.comments.list()
                print(f"🗨️  Found {len(comments)} comments")

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

            time.sleep(1)

        if found_old_post:
            break

    # --- SAVE RAW ---
    if all_entries:
        new_df = pd.DataFrame(all_entries)
        combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["text", "created"])
        combined.to_csv(CSV_PATH, index=False)
        print(f"\n✅ Added {len(new_df)} new entries. Total now: {len(combined)}")
    else:
        print("\n✅ No new posts or comments to add.")
        new_df = pd.DataFrame()
        combined = existing_df

    # --- FILTER NEW COMMENTS ONLY ---
    new_comments_df = new_df[new_df['type'] == 'comment'][['text', 'created']].drop_duplicates().reset_index(drop=True)
    new_comments_df['date'] = pd.to_datetime(new_comments_df['created']).dt.date

    # --- LOAD EXISTING SENTIMENTS ---
    if os.path.exists(OUTPUT_LABELED_PATH):
        labeled_df = pd.read_csv(OUTPUT_LABELED_PATH)
        if 'date' not in labeled_df.columns:
            labeled_df['date'] = pd.NaT
        print(f"📂 Loaded {len(labeled_df)} previously labeled comments.")
    else:
        labeled_df = pd.DataFrame(columns=["text", "sentiment", "date"])
        print("🆕 No labeled comments found, creating new file.")

    # --- FILTER COMMENTS NOT YET LABELED ---
    comments_to_label = new_comments_df[~new_comments_df['text'].isin(labeled_df['text'])].reset_index(drop=True)

    if comments_to_label.empty:
        print("ℹ️ No new unlabeled comments found.")
    else:
        # --- INIT CLASSIFIER ---
        print("\n🔬 Loading sentiment classifier...")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        classifier = pipeline("sentiment-analysis", model=model_name)

        label_map = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }

        # --- BATCH SENTIMENT WITH PROGRESS ---
        print("\n🧠 Labeling new comments (with progress bar)...")
        texts = comments_to_label['text'].str.slice(0, 512).tolist()
        sentiments = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size), desc="🔄 Classifying"):
            batch = texts[i:i + batch_size]
            try:
                results = classifier(batch, truncation=True)
                sentiments.extend([label_map.get(r['label'], 'unknown') for r in results])
            except Exception as e:
                print(f"⚠️ Error in batch {i}-{i+batch_size}: {e}")
                sentiments.extend(['error'] * len(batch))

        comments_to_label['sentiment'] = sentiments

        # --- APPEND NEW TO LABELED FILE ---
        final_labeled = pd.concat(
            [labeled_df, comments_to_label[['text', 'sentiment', 'date']]],
            ignore_index=True
        ).drop_duplicates(subset=['text'])

        final_labeled.to_csv(OUTPUT_LABELED_PATH, index=False)
        print(f"\n✅ Appended {len(comments_to_label)} new labeled comments to '{OUTPUT_LABELED_PATH}'")

import tweepy
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
bearer_token = os.getenv("BEARER_TOKEN")

if not bearer_token:
    raise ValueError("Bearer token not found. Check your .env file.")

# Authenticate with Twitter API
client = tweepy.Client(bearer_token=bearer_token)

# Fetch tweets from @TruthGundlach
username = 'TruthGundlach'
user = client.get_user(username=username)

if not user or not user.data:
    raise ValueError("Failed to fetch user. Check username or API limits.")

user_id = user.data.id

# Fetch recent tweets (max 100 from past 7 days)
tweets = client.get_users_tweets(
    id=user_id,
    max_results=100,
    tweet_fields=['created_at', 'public_metrics']
)

data = []

if tweets and tweets.data:
    for tweet in tweets.data:
        data.append({
            'text': tweet.text,
            'created_at': tweet.created_at,
            'likes': tweet.public_metrics['like_count'],
            'retweets': tweet.public_metrics['retweet_count']
        })
else:
    print("No tweets found or API limit reached.")

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('csvs/gundlach_tweets.csv', index=False)
print("Done! Saved to gundlach_tweets.csv")

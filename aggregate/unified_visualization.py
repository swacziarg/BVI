import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# --- Ensure output directory exists ---
os.makedirs("aggregate/visualizations", exist_ok=True)

# === Load and process Twitter ===
twitter = pd.read_csv("twitter/csvs/daily_sentiment_detail.csv", parse_dates=["date"])
twitter["date"] = pd.to_datetime(twitter["date"]).dt.tz_localize(None)
twitter.set_index("date", inplace=True)
twitter["platform"] = "Twitter"
twitter_summary = twitter[["pos_minus_neg", "platform"]].copy()

# === Load and process RSS ===
rss_raw = pd.read_csv("rss/csvs/rss_sentiment_results.csv", parse_dates=["published"])
rss_raw["date"] = pd.to_datetime(rss_raw["published"]).dt.tz_localize(None).dt.date
rss_raw["date"] = pd.to_datetime(rss_raw["date"])
rss_grouped = rss_raw.groupby(["date", "sentiment_label"]).size().unstack(fill_value=0)
for label in ["positive", "negative", "neutral"]:
    if label not in rss_grouped.columns:
        rss_grouped[label] = 0
rss_grouped["pos_minus_neg"] = rss_grouped["positive"] - rss_grouped["negative"]
rss_grouped["platform"] = "RSS"
rss_summary = rss_grouped[["pos_minus_neg", "platform"]].copy()
rss_summary.index.name = "date"

# === Load and process Reddit ===
reddit = pd.read_csv("reddit/csvs/reddit_sentiment_results.csv", parse_dates=["created"])

# Simulate sentiment columns if missing
if "sentiment_score" not in reddit.columns:
    # Fallback: compute sentiment score from label
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    reddit["sentiment_score"] = reddit["sentiment"].map(sentiment_map)

if "avg_comment_sentiment" not in reddit.columns:
    reddit["avg_comment_sentiment"] = reddit["sentiment_score"]  # fallback

reddit["date"] = pd.to_datetime(reddit["created"]).dt.tz_localize(None).dt.date
reddit["date"] = pd.to_datetime(reddit["date"])
reddit_grouped = reddit.groupby("date")[["sentiment_score", "avg_comment_sentiment"]].mean()
reddit_grouped["pos_minus_neg"] = reddit_grouped["sentiment_score"] - reddit_grouped["avg_comment_sentiment"]
reddit_grouped["platform"] = "Reddit"
reddit_summary = reddit_grouped[["pos_minus_neg", "platform"]].copy()
reddit_summary.index.name = "date"

# === Combine all ===
combined = pd.concat([twitter_summary, rss_summary, reddit_summary])
combined.sort_index(inplace=True)

# Limit to past 12 months
cutoff_date = combined.index.max() - pd.DateOffset(days=365)
combined = combined.loc[combined.index >= cutoff_date]

# --- Normalize sentiment scores ---
def normalize(group):
    return (group - group.mean()) / group.std()

combined["normalized_sentiment"] = combined.groupby("platform")["pos_minus_neg"].transform(normalize)

# --- Rolling 7-day trend ---
combined["trend"] = combined.groupby("platform")["normalized_sentiment"].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# === Load ETF data ===
cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
agg = pd.read_csv("aggregate/AGG.csv", names=cols, skiprows=1)
tlt = pd.read_csv("aggregate/TLT.csv", names=cols, skiprows=1)

agg = agg[["Date", "Close"]].rename(columns={"Close": "AGG"})
tlt = tlt[["Date", "Close"]].rename(columns={"Close": "TLT"})
agg["Date"] = pd.to_datetime(agg["Date"])
tlt["Date"] = pd.to_datetime(tlt["Date"])

etfs = pd.merge(agg, tlt, on="Date", how="inner").sort_values("Date")
etfs.set_index("Date", inplace=True)

# --- Align ETFs to sentiment dates ---
etfs = etfs.loc[combined.index.min():combined.index.max()]
etf_normalized = (etfs - etfs.mean()) / etfs.std()
etf_normalized = etf_normalized.reindex(combined.index).ffill()

# === Plot: Sentiment Trends and ETFs ===
plt.figure(figsize=(15, 6))

# Plot sentiment by platform
for platform in combined["platform"].unique():
    platform_data = combined[combined["platform"] == platform]
    plt.plot(platform_data.index, platform_data["trend"], label=f"{platform} (7d Avg)", linewidth=2)

# Overlay ETF performance
for ticker in etf_normalized.columns:
    plt.plot(etf_normalized.index, etf_normalized[ticker],
             linestyle="--", linewidth=1.5, label=f"{ticker} ETF (Z-Score)")

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title("Normalized Bond Sentiment vs. ETF Prices (Last 12 Months)", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Z-Score Normalized Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("aggregate/visualizations/bond_sentiment_vs_etfs.png")

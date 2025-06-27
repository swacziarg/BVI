import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Ensure output directory exists ===
os.makedirs("reddit/visualizations", exist_ok=True)

# === Load Reddit sentiment data ===
df = pd.read_csv("reddit/csvs/auto_labeled_reddit_comments.csv")

# --- Preprocessing ---

# Simulate 'created' datetime if missing
if 'created' not in df.columns:
    df['created'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="h")

# Simulate 'type' column if missing
if 'type' not in df.columns:
    df['type'] = 'comment'

# Validate sentiment values
valid_sentiments = {"positive", "neutral", "negative"}
df = df[df["sentiment"].isin(valid_sentiments)]

# Convert dates
df.rename(columns={"created": "datetime"}, inplace=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.date
df["date"] = pd.to_datetime(df["date"])

# Map sentiment to numeric scores
sentiment_to_score = {"positive": 1, "neutral": 0, "negative": -1}
df["sentiment_score"] = df["sentiment"].map(sentiment_to_score)

# --- Daily sentiment aggregation ---
daily_sentiment = df.groupby(["date", "type"])["sentiment_score"].mean().unstack(fill_value=0)
daily_sentiment["combined"] = daily_sentiment.mean(axis=1)

# --- Ensure full daily index ---
daily_sentiment = daily_sentiment.asfreq("D").fillna(0)

# === Load ETF data ===
etf_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
agg = pd.read_csv("aggregate/AGG.csv", names=etf_cols, skiprows=1)
tlt = pd.read_csv("aggregate/TLT.csv", names=etf_cols, skiprows=1)

agg = agg[["Date", "Close"]].rename(columns={"Close": "AGG"})
tlt = tlt[["Date", "Close"]].rename(columns={"Close": "TLT"})

agg["Date"] = pd.to_datetime(agg["Date"])
tlt["Date"] = pd.to_datetime(tlt["Date"])

etfs = pd.merge(agg, tlt, on="Date", how="inner").sort_values("Date").set_index("Date")

# --- Align date range to ETFs ---
etf_start = etfs.index.min()
sentiment_end = daily_sentiment.index.max()
date_range = pd.date_range(etf_start, sentiment_end, freq='D')

# Clip and reindex sentiment to ETF-aligned range
daily_sentiment = daily_sentiment.loc[etf_start:sentiment_end].reindex(date_range).fillna(0)

# Reindex ETF prices to match sentiment
etfs = etfs.reindex(date_range).ffill()

# --- Normalize (Z-score) ---
etfs_normalized = (etfs - etfs.mean()) / etfs.std()
sentiment_normalized = (daily_sentiment["combined"] - daily_sentiment["combined"].mean()) / daily_sentiment["combined"].std()

# --- 30-day rolling sentiment ---
rolling_sentiment = sentiment_normalized.rolling(window=7, min_periods=1).mean()

# === Plot: Normalized Reddit Sentiment vs ETF Prices ===
plt.figure(figsize=(15, 6))

# Plot smoothed sentiment
plt.plot(rolling_sentiment.index, rolling_sentiment,
         label="Reddit Sentiment (7d Avg, Z-Score)", linewidth=2, color='navy')

# Plot ETF normalized prices
for ticker in etfs_normalized.columns:
    plt.plot(etfs_normalized.index, etfs_normalized[ticker],
             label=f"{ticker} ETF (Normalized)", linestyle='--', linewidth=1.8)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title("Normalized Reddit Sentiment vs Bond ETF Prices", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Z-Score Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reddit/visualizations/reddit_sentiment_vs_etfs_normalized.png")

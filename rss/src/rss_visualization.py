import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Load and preprocess data ---
rss = pd.read_csv("rss/csvs/rss_sentiment_results.csv", parse_dates=["published"])
rss["date"] = pd.to_datetime(rss["published"]).dt.tz_localize(None).dt.date
rss["date"] = pd.to_datetime(rss["date"])

# --- Create daily sentiment counts ---
daily_counts = df.groupby(["date", "sentiment_label"]).size().unstack(fill_value=0)

# Ensure all sentiment types are present
for label in ["positive", "negative", "neutral"]:
    if label not in daily_counts.columns:
        daily_counts[label] = 0

# --- Calculate net sentiment score ---
daily_counts["pos_minus_neg"] = daily_counts["positive"] - daily_counts["negative"]
daily_counts.index = pd.to_datetime(daily_counts.index)

# --- Rolling trend (7-day average) ---
daily_counts['sentiment_trend'] = daily_counts['pos_minus_neg'].rolling(window=7, min_periods=1).mean()

# === Plot 1: Full Time Series ===
plt.figure(figsize=(14, 6))
plt.plot(daily_counts.index, daily_counts['sentiment_trend'], label='Bond Sentiment (7d Avg)', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Bond Market Sentiment Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Net Sentiment (Positive - Negative)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rss/visualizations/bond_sentiment_plot_full.png")
 
# === Plot 2: Past 7 Days ===
last_date = daily_counts.index.max()
start_date = last_date - timedelta(days=6)
weekly_data = daily_counts.loc[start_date:last_date]

plt.figure(figsize=(10, 4))
plt.plot(weekly_data.index, weekly_data['pos_minus_neg'], marker='o', linestyle='-', label='Daily Net Sentiment')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Bond Market Sentiment – Past 7 Days", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Net Sentiment")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rss/visualizations/bond_sentiment_plot_week.png")

# === Plot 3: Cumulative Sentiment (Accumulator) ===
daily_counts['sentiment_accumulator'] = daily_counts['pos_minus_neg'].cumsum()

plt.figure(figsize=(14, 5))
plt.plot(daily_counts.index, daily_counts['sentiment_accumulator'], color='purple', linewidth=2, label='Cumulative Sentiment')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Cumulative Bond Market Sentiment Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Cumulative Sentiment (Sum of Net Daily Sentiment)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rss/visualizations/bond_sentiment_accumulator.png")

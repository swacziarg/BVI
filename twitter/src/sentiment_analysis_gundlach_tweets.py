import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load tweets
df = pd.read_csv("twitter/csvs/gundlach_tweets.csv")
df['created_at'] = pd.to_datetime(df['created_at'])
df['date'] = df['created_at'].dt.normalize()

# Expanded bond-related keywords
bond_keywords = [
    # Instruments
    "bond", "bonds", "treasury", "treasuries", "t-bond", "t-note", "t-bill", "muni", "municipal bond",
    "junk bond", "high yield", "corporate bond", "sovereign debt", "fixed income", "long bond", "short duration",
    "10-year", "2-year", "30-year", "convertible bond", "callable bond",

    # Rates & yields
    "yield", "yields", "interest rate", "rates", "fed funds", "fed funds rate", "curve", "yield curve", "inversion",
    "curve inversion", "spread", "credit spread", "basis points", "bps", "zero coupon", "real yield", "nominal yield",

    # Central bank / monetary policy
    "fed", "federal reserve", "powell", "fomc", "quantitative easing", "qe", "tapering", "rate hike", "rate cut",
    "rate pause", "tightening", "easing", "monetary policy", "dual mandate",

    # Macro/fiscal context
    "inflation", "cpi", "ppi", "stagflation", "recession", "gdp", "budget deficit", "deficit", "debt",
    "debt ceiling", "debt market", "fiscal policy", "spending", "sovereign risk", "coupon",

    # Market behavior / sentiment
    "vigilante", "bond vigilante", "hawkish", "dovish", "default", "downgrade", "bankrupt", "liquidity",
    "volatility", "duration", "price discovery", "macro", "macroeconomics", "treasury auction", "issuance",
    "rollover", "credit event",

    # People / commentary / misc
    "gundlach", "gross", "doubleline", "pimco", "cnbc", "bloomberg", "webcast", "outlook", "spread widening",
    "carry trade", "cut", "economy"
]

def is_bond_related(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in bond_keywords)

df['is_bond_related'] = df['text'].apply(is_bond_related)
df_bonds = df[df['is_bond_related']].copy()
df_bonds.to_csv("twitter/csvs/filtered_bond_tweets.csv", index=False)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Classify sentiment
def classify(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Override VADER score for sarcastic or concern-heavy finance cases
def override_vader(text, score):
    text_lower = text.lower()
    if (
        "debt clock" in text_lower or
        ("trillion" in text_lower and "debt" in text_lower) or
        ("trillion" in text_lower and "yippee" in text_lower) or
        "debt ceiling" in text_lower or
        "fiscal cliff" in text_lower
    ):
        return -0.5  # strong negative override
    return score

# Score and classify sentiment with override
df_bonds['sentiment_score'] = df_bonds['text'].apply(
    lambda x: override_vader(x, analyzer.polarity_scores(x)['compound'])
)
df_bonds['sentiment_label'] = df_bonds['sentiment_score'].apply(classify)


# Group by date and sentiment
daily_counts = df_bonds.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)

# Ensure all sentiment types exist
for col in ['positive', 'neutral', 'negative']:
    if col not in daily_counts.columns:
        daily_counts[col] = 0

# Add totals
daily_counts['total_tweets'] = daily_counts['positive'] + daily_counts['neutral'] + daily_counts['negative']
daily_counts['pos_minus_neg'] = daily_counts['positive'] - daily_counts['negative']

# Reindex to fill in missing days
date_range = pd.date_range(start=df_bonds['date'].min(), end=df_bonds['date'].max(), freq='D')
daily_counts = daily_counts.reindex(date_range, fill_value=0)
daily_counts.index.name = 'date'

# Compute deltas
def safe_diff(col):
    return col.diff().fillna(0).astype(int)

daily_counts['Δ_total'] = safe_diff(daily_counts['total_tweets'])
daily_counts['Δ_pos'] = safe_diff(daily_counts['positive'])
daily_counts['Δ_neg'] = safe_diff(daily_counts['negative'])
daily_counts['Δ_neutral'] = safe_diff(daily_counts['neutral'])
daily_counts['Δ_pos_minus_neg'] = safe_diff(daily_counts['pos_minus_neg'])

# Output
summary = daily_counts.reset_index()
summary.to_csv("twitter/csvs/daily_sentiment_detail.csv", index=False)
print(summary.head(10))

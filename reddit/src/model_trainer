import pandas as pd
import sys
import termios
import tty
import os

# --- Load Reddit CSV ---
reddit_df = pd.read_csv("reddit/csvs/reddit_raw_posts_and_comments.csv")
if 'text' not in reddit_df.columns or 'type' not in reddit_df.columns:
    raise ValueError("CSV must contain 'text' and 'type' columns")

# Filter only comments
comments_df = reddit_df[reddit_df['type'] == 'comment'][['text']].drop_duplicates().reset_index(drop=True)

# --- Helper to read one char without Enter ---
def read_single_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# --- Label comments interactively ---
labels = []
output_path = "labeled_reddit_comments.csv"
print("\nLabel each comment:\n 1 = Positive, 2 = Negative, 3 = Neutral, 4 = Skip\nPress Ctrl+C to stop anytime.\n")

try:
    for i, row in comments_df.iterrows():
        os.system("clear" if os.name == "posix" else "cls")
        print(f"\nComment #{i+1} of {len(comments_df)}:\n")
        print(row['text'])
        print("\n[1] Positive  [2] Negative  [3] Neutral  [4] Skip")

        while True:
            key = read_single_keypress()
            if key in {'1', '2', '3', '4'}:
                break
            else:
                print("\nInvalid input. Press 1, 2, 3, or 4:")

        if key == '4':
            continue  # Skip

        sentiment = {'1': 'positive', '2': 'negative', '3': 'neutral'}[key]
        labels.append({'text': row['text'], 'sentiment': sentiment})
except KeyboardInterrupt:
    print("\n\nStopped labeling by user.")

# --- Save results ---
if labels:
    df_out = pd.DataFrame(labels)
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df_out)} labeled comments to '{output_path}'")
else:
    print("\n⚠️ No comments labeled.")

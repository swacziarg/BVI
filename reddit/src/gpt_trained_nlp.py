if __name__ == "__main__":
    import pandas as pd
    from transformers import pipeline

    # --- Load your Reddit CSV ---
    df = pd.read_csv("reddit/csvs/reddit_raw_posts_and_comments.csv")

    # --- Filter only comments ---
    comments_df = df[df['type'] == 'comment'][['text']].drop_duplicates().reset_index(drop=True)

    # --- Load sentiment classifier ---
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    classifier = pipeline("sentiment-analysis", model=model_name)

    # --- Label mapping ---
    label_map = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }

    # --- Sentiment prediction function with truncation ---
    def classify(text):
        try:
            result = classifier(text[:512])[0]
            return label_map.get(result['label'], 'unknown')
        except Exception as e:
            print(f"⚠️ Error classifying comment: {e}")
            return 'error'

    # --- Apply classification ---
    print("🔍 Labeling comments... This may take a few minutes.")
    comments_df['sentiment'] = comments_df['text'].apply(classify)

    # --- Save output ---
    output_path = "auto_labeled_reddit_comments.csv"
    comments_df.to_csv(output_path, index=False)
    print(f"✅ Done! Labeled comments saved to '{output_path}'")

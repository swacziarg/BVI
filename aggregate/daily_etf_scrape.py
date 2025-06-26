import yfinance as yf
import pandas as pd
from datetime import datetime
import time

TICKERS = ['aggregate/AGG', 'aggregate/TLT']

def fetch_historical_data_yf(ticker, days_back=30):
    symbol = ticker.split('/')[-1]
    try:
        df = yf.download(symbol, period=f"{days_back}d", auto_adjust=False, progress=False)
        if df.empty:
            print(f"{ticker}: Downloaded data is empty.")
            return pd.DataFrame()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Volume'] = df['Volume'].fillna(0).astype(int)
        return df
    except Exception as e:
        print(f"{ticker}: Failed to fetch data. Reason: {e}")
        return pd.DataFrame()

def update_and_get_new_rows(ticker):
    csv_file = f"{ticker}.csv"

    try:
        existing_df = pd.read_csv(csv_file)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%m/%d/%Y')
        last_date = existing_df['Date'].max()
    except FileNotFoundError:
        print(f"{csv_file} not found. Creating new file.")
        existing_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        last_date = None

    new_data_df = fetch_historical_data_yf(ticker)
    
    if new_data_df.empty or 'Date' not in new_data_df.columns:
        print(f"{ticker}: No new data retrieved.")
        return pd.DataFrame()

    print(f"Latest scraped dates for {ticker}:")
    print(new_data_df['Date'].head().dt.strftime('%m/%d/%Y').to_list())

    if last_date:
        new_rows = new_data_df[new_data_df['Date'] > last_date]
    else:
        new_rows = new_data_df

    if not new_rows.empty:
        updated_df = pd.concat([new_rows, existing_df], ignore_index=True)
        updated_df.sort_values('Date', ascending=False, inplace=True)
        updated_df['Date'] = updated_df['Date'].dt.strftime('%m/%d/%Y')
        updated_df.to_csv(csv_file, index=False)
        print(f"{ticker}: Added {len(new_rows)} new rows:")
        print(new_rows.to_string(index=False))
    else:
        print(f"{ticker}: No new data since {last_date.strftime('%m/%d/%Y') if last_date else 'start'}.")

    return new_rows

# === Run for all tickers ===
for ticker in TICKERS:
    print(f"\n=== {ticker} ===")
    update_and_get_new_rows(ticker)
    time.sleep(10)  # Prevent Yahoo Finance rate limits

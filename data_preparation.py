import yfinance as yf
import pandas as pd

def get_market_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

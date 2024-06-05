import pandas as pd
from textblob import TextBlob
import numpy as np

def sentiment_analysis(tweets):
    sentiments = [TextBlob(tweet).sentiment.polarity for tweet in tweets]
    return np.mean(sentiments)

def trade_signal(row, model, sentiment, feature_names):
    momentum_signal = 1 if row['Close'] > row['MA50'] else -1
    features = pd.DataFrame([row[feature_names].values], columns=feature_names).values  # 使用 .values 只保留数据
    classification_signal = model.predict(features)[0]
    sentiment_signal = 1 if sentiment > 0 else -1
    combined_signal = (0.4 * momentum_signal) + (0.4 * classification_signal) + (0.2 * sentiment_signal)
    return combined_signal

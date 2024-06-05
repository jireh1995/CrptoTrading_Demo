from sentiment import sentiment_analysis, trade_signal  # Ensure these are correctly imported

def backtest(data, model, tweets, initial_balance, feature_names):
    balance = initial_balance
    position = 0
    returns = []

    print("Starting backtest loop...")
    for i, (index, row) in enumerate(data.iterrows()):
        sentiment = sentiment_analysis(tweets)
        signal = trade_signal(row, model, sentiment, feature_names)
        
        if signal > 0 and balance > 0:
            buy_amount = balance * 0.1 / row['Close']
            balance -= buy_amount * row['Close']
            position += buy_amount
        elif signal < 0 and position > 0:
            sell_amount = position * 0.1
            balance += sell_amount * row['Close']
            position -= sell_amount
        
        total_value = balance + position * row['Close']
        returns.append(total_value)
        
        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Iteration {i}: Total Value = {total_value}")

    print("Backtest loop completed.")
    data['Total Value'] = returns
    data['Returns'] = data['Total Value'].pct_change()
    
    return data

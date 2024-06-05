from data_preparation import get_market_data
from model_training import train_model
from backtest import backtest
from sentiment import sentiment_analysis, trade_signal  # Ensure these are correctly imported

if __name__ == "__main__":
    # 用户输入初始资产总额
    initial_balance = float(input("请输入初始资产总额: "))
    
    print("Fetching market data...")
    # 获取市场数据
    data = get_market_data('ETH-USD', '2018-01-01', '2024-06-05')
    print("Market data fetched.")
    
    print("Training model...")
    # 训练模型
    model, feature_names = train_model(data)
    print("Model trained.")
    
    # 模拟推文数据
    tweets = ["Sample tweet about Ethereum"] * len(data)
    
    print("Running backtest...")
    # 运行回测
    backtest_result = backtest(data, model, tweets, initial_balance, feature_names)
    print("Backtest completed.")
    
    # 输出结果
    final_balance = backtest_result['Total Value'].iloc[-1]
    total_return = (final_balance - initial_balance) / initial_balance * 100
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")

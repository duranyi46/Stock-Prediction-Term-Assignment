import os
import pandas as pd
import numpy as np

def backtest_top3_with_sharpe(
    predictions_df, data, unique_stocks, initial_balance=100000, action_fee=0.001,
    risk_free_rate=0.0, take_profit=0.04):
    
    total_cash = initial_balance
    holdings = {stock: 0 for stock in unique_stocks}
    purchase_prices = {stock: 0 for stock in unique_stocks}
    actions = []  # List to store actions
    trade_returns = []  # List to store trade returns

    predictions_df = predictions_df.sort_values(by='date')

    for date in predictions_df['date'].unique():
        daily_predictions = predictions_df[predictions_df['date'] == date]
        top_3_stocks = daily_predictions.nlargest(5, 'predicted_return')['stock_name'].tolist()

        for stock in unique_stocks:
            stock_data = data[data['stock_name'] == stock].reset_index(drop=True)
            current_day_data = stock_data[stock_data['date'] == date]

            if len(current_day_data) == 0 or stock not in top_3_stocks:
                continue

            current_price = current_day_data.iloc[0]['adj_close']
            next_day_data = stock_data[stock_data['date'] > date].iloc[:1]
            if next_day_data.empty:
                continue

            next_open_price = next_day_data.iloc[0]['open']

            # Buy logic
            if stock in top_3_stocks and holdings[stock] == 0:
                investment_amount = total_cash / 3
                shares_to_buy = max(1, investment_amount // next_open_price)

                if shares_to_buy > 0:
                    total_cash -= shares_to_buy * next_open_price * (1 + action_fee)
                    holdings[stock] += shares_to_buy
                    purchase_prices[stock] = next_open_price
                    actions.append({
                        'signal_date': date,
                        'trade_date': next_day_data.iloc[0]['date'],
                        'stock': stock,
                        'action': 'buy',
                        'price': next_open_price,
                        'shares': shares_to_buy,
                        'total_cash': total_cash,
                        'portfolio_value': total_cash + sum(
                            holdings[s] * stock_data[stock_data['date'] <= date].iloc[-1]['close']
                            for s in unique_stocks
                        ),
                        'profit_percentage': 0  # No profit percentage on buy
                    })

            # Sell logic with take profit and stop loss
            if holdings[stock] > 0:
                # Calculate return
                trade_return = (next_open_price - purchase_prices[stock]) / purchase_prices[stock]

                # Check thresholds
                if trade_return >= take_profit:
                    total_cash += holdings[stock] * next_open_price * (1 - action_fee)
                    trade_returns.append(trade_return)
                    actions.append({
                        'signal_date': date,
                        'trade_date': next_day_data.iloc[0]['date'],
                        'stock': stock,
                        'action': 'sell',
                        'price': next_open_price,
                        'shares': holdings[stock],
                        'total_cash': total_cash,
                        'portfolio_value': total_cash + sum(
                            holdings[s] * stock_data[stock_data['date'] <= date].iloc[-1]['close']
                            for s in unique_stocks
                        ),
                        'profit_percentage': trade_return * 100
                    })
                    holdings[stock] = 0  # Reset holdings

    # Convert actions to a DataFrame
    actions_df = pd.DataFrame(actions)

    # Calculate final portfolio value
    final_portfolio_value = total_cash + sum(
        holdings[stock] * data[data['stock_name'] == stock].iloc[-1]['close']
        for stock in unique_stocks
    )
    net_profit = final_portfolio_value - initial_balance
    roi = (net_profit / initial_balance) * 100

    # Calculate Sharpe Ratio
    if trade_returns:
        average_return = np.mean(trade_returns)
        std_dev_return = np.std(trade_returns)
        sharpe_ratio = (average_return - risk_free_rate) / (std_dev_return + 1e-8)
    else:
        sharpe_ratio = 0

    return final_portfolio_value, net_profit, roi, sharpe_ratio, actions_df
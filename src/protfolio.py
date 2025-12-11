import numpy as np
import pandas as pd

def portfolio_returns(weights, returns):
    return returns.mul(weights, axis=1).sum(axis=1)

def sharpe_ratio(returns):
    return (returns.mean() / returns.std()) * np.sqrt(252)

def max_drawdown(cumulative):
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def backtest(weights, stock_returns):
    daily = portfolio_returns(weights, stock_returns)
    cumulative = (1 + daily).cumprod()
    sharpe = sharpe_ratio(daily)
    mdd = max_drawdown(cumulative)
    return daily, cumulative, sharpe, mdd

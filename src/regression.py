import pandas as pd
import statsmodels.api as sm

def run_ols(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params

def compute_all_betas(returns, factors):
    betas = {}
    for stock in returns.columns:
        params = run_ols(returns[stock], factors)
        betas[stock] = params
    return pd.DataFrame(betas).T
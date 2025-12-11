import pandas as pd
import statsmodels.api as sm

def rolling_beta(y, X, window=126):
    betas = []
    for i in range(window, len(y)):
        y_win = y.iloc[i-window:i]
        X_win = sm.add_constant(X.iloc[i-window:i])
        model = sm.OLS(y_win, X_win).fit()
        betas.append(model.params.values)
    
    columns = ["const"] + list(X.columns)
    index = y.index[window:]
    return pd.DataFrame(betas, index=index, columns=columns)

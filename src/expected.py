import numpy as np
import pandas as pd

def compute_expected_return(beta_row, factor_premiums):
    return beta_row["const"] + np.dot(beta_row[1:], factor_premiums)

def compute_all_expected_returns(betas, premiums):
    expected = {}
    for stock in betas.index:
        expected[stock] = compute_expected_return(betas.loc[stock], premiums)
    return pd.Series(expected).sort_values(ascending=False)

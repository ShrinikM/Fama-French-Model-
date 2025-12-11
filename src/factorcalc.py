import pandas as pd

def convert_to_decimal(df):
    for col in df.columns:
        df[col] = df[col] / 100
    return df

def compute_factor_premiums(factors):
    daily_prem = factors.mean()
    annual = (1 + daily_prem)**252 - 1
    return annual

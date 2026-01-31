import pandas as pd
import numpy as np
import datetime as dt

from src.loader import download_prices, load_fama_french, merge_stock_factors
from src.factorcalc import convert_to_decimal, compute_factor_premiums
from src.regression import compute_all_betas
from src.expected import compute_all_expected_returns
from src.protfolio import backtest


TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META",
    "TSLA", "AMZN", "JPM", "BAC", "XOM",
    "CVX", "WMT", "TGT", "HD", "NKE",
    "MCD", "KO", "PEP", "ORCL", "IBM",
    "CSCO", "QCOM", "INTC", "AVGO", "ADBE",
    "NFLX", "AMD", "CRM", "PYPL", "ABNB",
    "UBER", "LYFT", "SPY", "CAT", "UPS",
    "LOW", "COST", "GS", "MS", "GE",
    "F", "GM", "UNH", "PFE", "MRK",
    "BA", "LMT", "SBUX", "DIS", "BK"
]

START = "2019-01-01"
END = "2025-01-01"

FF_PATH = "./data/raw/fama_french_factors.csv"  

print("\n=== FACTOR MODEL PIPELINE STARTED ===\n")



print("[1/9] Downloading stock data...")

prices = download_prices(TICKERS, START, END)
stock_returns = prices.pct_change().dropna()

print("✔ Stock data downloaded and converted to returns.")



print("[2/9] Loading Fama–French factor dataset...")

ff = load_fama_french(FF_PATH)
ff = convert_to_decimal(ff)  # convert % to decimal
ff = ff.rename(columns={"mkt_rf": "Market"})  

print("✔ Fama–French data loaded and cleaned.")

print("[3/9] Merging stock returns with factor data...")

merged = merge_stock_factors(stock_returns, ff)
merged.to_csv("./data/processed/merged_dataset.csv")

print("✔ Merged dataset saved.")



# ff = load_fama_french(FF_PATH)
# print("FF Columns:", ff.columns.tolist())
# print(ff.head())
# exit()

# factor_cols = ["Market", "SMB", "HML", "RMW", "CMA"]
# print("Merged columns:", merged.columns.tolist())  # DEBUG
# factors = merged[factor_cols]

print("[4/9] Running factor regressions (OLS)...")

factor_cols = ["Market", "SMB", "HML", "RMW", "CMA"]
factors = merged[factor_cols]
returns = merged[TICKERS]

betas = compute_all_betas(returns, factors)
betas.to_csv("./data/processed/factor_betas.csv")

print("✔ Betas computed and saved.")
print("[5/9] Computing factor premiums and expected returns...")

factor_premiums = compute_factor_premiums(factors)
expected_returns = compute_all_expected_returns(betas, factor_premiums)
expected_returns.to_csv("./results/expected_returns.csv")

print("✔ Expected returns saved.")
print("[6/9] Building long-only portfolio...")

top20 = expected_returns.head(20)
weights_lo = np.ones(len(top20)) / len(top20)
weights_lo = pd.Series(weights_lo, index=top20.index)

print("✔ Long-only portfolio constructed.")

print("[7/9] Running backtest...")

daily_lo, cumulative_lo, sharpe_lo, mdd_lo = backtest(weights_lo, returns[top20.index])

summary = pd.DataFrame({
    "Annualized Return": [(1 + daily_lo.mean())**252 - 1],
    "Sharpe Ratio": [sharpe_lo],
    "Max Drawdown": [mdd_lo]
})

summary.to_csv("./results/backtest_summary.csv")
cumulative_lo.to_csv("./results/cumulative_portfolio.csv")

print("Backtest complete.")
print(summary)
print("\n=== PIPELINE COMPLETE ===")

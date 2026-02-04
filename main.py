import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from src.loader import download_prices, load_fama_french, merge_stock_factors
from src.factorcalc import convert_to_decimal, compute_factor_premiums
from src.regression import compute_all_betas
from src.expected import compute_all_expected_returns
from src.protfolio import backtest_ols
from src.random_forest import FactorRandomForest


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

TOP_N = 10
REBALANCE_MONTHS = 12
INITIAL_CAPITAL = 10000



def process_data_for_ml(prices, factors):
    """Helper to convert Daily Data -> Monthly Long Format for ML"""
    print("   [Info] Processing data for ML...")

    prices_mo = prices.resample('ME').last()
    returns_mo = prices_mo.pct_change().dropna()
    factors_mo = factors.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    if returns_mo.index.tz is not None:
        returns_mo.index = returns_mo.index.tz_localize(None)
    if factors_mo.index.tz is not None:
        factors_mo.index = factors_mo.index.tz_localize(None)
        
    merged = returns_mo.join(factors_mo, how='inner')
    
    print(f"   [Debug] ML Data Merge Shape: {merged.shape}")
    if merged.empty:
        print("   [ERROR] Merge resulted in empty DataFrame! Check Date alignment.")
        return pd.DataFrame()
    
    merged.index.name = 'date'
    factor_cols = list(factors_mo.columns)
    
    return merged.reset_index().melt(id_vars=['date'] + factor_cols, var_name='Ticker', value_name='Return')

def calculate_metrics(returns_series, name="Strategy"):
    """Standardizes metrics for comparison"""
    ann_ret = returns_series.mean() * 12
    ann_vol = returns_series.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    cum = (1 + returns_series).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    return {
        "Strategy": name,
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

if __name__ == "__main__":
    print("\n=== QUANT PIPELINE STARTED ===\n")
    print("[1/3] Loading Shared Data...")
    prices = download_prices(TICKERS, START, END)
    stock_returns = prices.pct_change().dropna()
    
    ff = load_fama_french(FF_PATH)
    ff = convert_to_decimal(ff)
    
    ff = ff.rename(columns={"mkt_rf": "Market"})
    
    merged_daily = merge_stock_factors(stock_returns, ff)
    merged_daily.to_csv("./data/processed/merged_dataset.csv")
    print("✔ Data Loaded & Merged.")

    print("\n--- PHASE 1: Running OLS Regression Model ---")
    
    factor_cols = ["Market", "SMB", "HML", "RMW", "CMA"]
    factors = merged_daily[factor_cols]
    returns = merged_daily[TICKERS]

    betas = compute_all_betas(returns, factors)
    betas.to_csv("./data/processed/factor_betas.csv")
    
    premiums = compute_factor_premiums(factors)
    exp_returns_ols = compute_all_expected_returns(betas, premiums)
    
    top20_ols = exp_returns_ols.head(20)
    weights_ols = pd.Series(np.ones(len(top20_ols)) / len(top20_ols), index=top20_ols.index)
    
    daily_ols, cum_ols, sharpe_ols, mdd_ols = backtest_ols(weights_ols, returns[top20_ols.index])
    print(f"✔ OLS Strategy Complete. Sharpe: {sharpe_ols:.2f}")
    print("\n--- PHASE 2: Running Random Forest ML Model ---")
    
    ff_ml = ff.rename(columns={"Market": "MKT_RF"})
    
    df_ml = process_data_for_ml(prices, ff_ml)
    
    if df_ml.empty:
        print("[CRITICAL ERROR] ML Data Processing Failed. Stopping.")
        exit()

    rf_model = FactorRandomForest(n_estimators=100, max_depth=5)
    data_ml = rf_model.prepare_data(df_ml)
    
    unique_dates = sorted(data_ml['date'].unique())
    
    if len(unique_dates) < 24:
        print(f"   [ERROR] Not enough history for ML (Found {len(unique_dates)} months, need 24+).")
        exit()
        
    start_idx = min(48, int(len(unique_dates) * 0.5))
    ml_returns = []
    
    print(f"   > Starting Rolling Training (Total Months: {len(unique_dates)}, Start Index: {start_idx})...")
    
    for i in range(start_idx, len(unique_dates) - 1):
        curr_date = unique_dates[i]
        next_date = unique_dates[i+1]
        
        if (i - start_idx) % REBALANCE_MONTHS == 0:
            print(f"     > Re-training model at {curr_date.date()}...")
            train_mask = data_ml['date'] <= curr_date
            rf_model.train(data_ml[train_mask])
            
        current_data = data_ml[data_ml['date'] == curr_date]
        if current_data.empty: continue
        
        preds = rf_model.predict(current_data)
        
        signals = current_data.copy()
        signals['Score'] = preds
        long_picks = signals.nlargest(TOP_N, 'Score')
        
        period_ret = long_picks['Next_Month_Return'].mean()
        
        ml_returns.append({'date': next_date, 'Return': period_ret})
        
    if not ml_returns:
        print("[ERROR] No predictions generated. Exiting.")
        exit()

    ml_results_df = pd.DataFrame(ml_returns).set_index('date')
    ml_series = ml_results_df['Return'] 
    
    print("\n" + "="*40)
    print("FINAL RESULTS: TRADITIONAL (OLS) vs. AI (ML)")
    print("="*40)
     
    metrics_ols = calculate_metrics(daily_ols, "OLS (Static)")
    metrics_ml = calculate_metrics(ml_series, "Random Forest (Dynamic)")
    
    comparison = pd.DataFrame([metrics_ols, metrics_ml]).set_index("Strategy")
    
    
    print(comparison.style.format({
        "Ann. Return": "{:.2%}",
        "Ann. Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}"
    }).to_string())
    
    comparison.to_csv("./results/strategy_comparison.csv")
    print("\n✔ Results saved to ./results/strategy_comparison.csv")
    plt.figure(figsize=(10, 5))
    (1 + daily_ols).cumprod().plot(label="OLS Static (Daily)", alpha=0.6)
    ml_cum = (1 + ml_series).cumprod()
    ml_cum.plot(label="Random Forest (Monthly)", linewidth=2)
    
    plt.title("OLS vs. Random Forest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./results/final_comparison_chart.png")
    print("✔ Comparison Chart saved.")
    
    print("\n=== PIPELINE COMPLETE ===")

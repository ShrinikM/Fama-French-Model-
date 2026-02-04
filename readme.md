# Multi-Factor Asset Pricing Model & Portfolio Backtester

A research pipeline implementing the Fama-French 5-Factor framework with integrated machine learning and backtesting.

## Overview

This project provides an end-to-end implementation of a factor-based investing model. It automates the workflow from data ingestion to performance evaluation, benchmarking a traditional linear model against a dynamic Random Forest approach.

The pipeline is fully automated via `main.py` and designed for modular extension.

## Data Pipeline
* **Automated Retrieval:** Fetches adjusted prices for 50+ stocks via `yfinance`.
* **Factor Processing:** Cleans and aligns Fama-French 5-factor data.
* **Feature Engineering:** Computes 1-month Momentum and Excess Returns ($R_i - R_f$).
* **Time-Series Sync:** Aligns disparate daily and monthly data using timezone-naive timestamps to prevent data leakage.

## Machine Learning Methodology
* **Dynamic Ranking:** Uses Random Forest to rank stocks by predicted "Alpha Scores."
* **Walk-Forward Training:** Re-trains annually to adapt to shifting market regimes.
* **Non-Linear Modeling:** Captures complex factor interactions missed by linear models.
* **Robust Selection:** XGBoost was tested but rejected ($R^2$: -0.17) in favor of Random Forest’s superior generalization (Sharpe: 1.91).

## Factor Regression (Baseline)
Implements OLS regression via `statsmodels` to estimate stock exposures to:
* Market Risk Premium ($β_M$)
* Size (SMB), Value (HML), Profitability (RMW), and Investment (CMA)

$$ER_i = \beta_{iM} \cdot RP_M + \beta_{iSMB} \cdot RP_{SMB} + \beta_{iHML} \cdot RP_{HML} + \beta_{iRMW} \cdot RP_{RMW} + \beta_{iCMA} \cdot RP_{CMA}$$

## Backtesting & Performance
Evaluates strategy performance using annualized returns, Sharpe ratio, and Max Drawdown.

| Metric | Traditional OLS (Baseline) | Random Forest (ML) |
| :--- | :--- | :--- |
| **Annualized Return** | 1.09% | **47.38%** |
| **Sharpe Ratio** | 0.24 | **1.91** |
| **Max Drawdown** | -36.50% | **-15.43%** |



## Setup & Usage
1. **Prerequisites:** Python 3.11
2. **Installation:** `pip install -r requirements.txt`
3. **Run:** `python main.py`

---
*Disclaimer: This project is for educational purposes only and does not constitute financial advice.*

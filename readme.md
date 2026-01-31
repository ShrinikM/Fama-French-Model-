# Multi-Factor Asset Pricing Model & Portfolio Backtester

A quantitative research pipeline implementing the Fama-French 5-Factor framework with integrated backtesting capabilities.

## Overview

This project provides an end-to-end implementation of a multi-factor investing model used in quantitative research environments. The system automates the complete workflow from data ingestion through performance evaluation:

- Data acquisition and preprocessing
- Factor regression analysis
- Beta estimation with rolling exposures
- Expected return modeling
- Portfolio construction
- Performance backtesting and analytics

The pipeline is fully automated via `main.py` and designed for production use and extension.

## Core Capabilities

**Data Pipeline**
- Retrieves historical equity prices for 50+ stocks using yfinance
- Processes Fama-French 5-factor data
- Handles return conversions and data alignment
- Merges factor and return datasets

**Factor Regression**
Implements OLS regression via statsmodels to estimate stock exposures to:
- Market risk premium
- SMB (Size factor)
- HML (Value factor)
- RMW (Profitability factor)
- CMA (Investment factor)

**Expected Return Modeling**
Calculates expected returns using factor loadings and historical factor premiums:

```
ER_i = β_iM · RP_M + β_iSMB · RP_SMB + β_iHML · RP_HML + β_iRMW · RP_RMW + β_iCMA · RP_CMA
```

**Portfolio Construction**
- Ranks securities by expected return
- Constructs equal-weighted long-only portfolios (top 20 holdings)
- Extensible to long/short, risk parity, and optimization-based strategies

**Backtesting Framework**
Evaluates strategy performance with standard metrics:
- Annualized returns
- Sharpe ratio
- Maximum drawdown
- Cumulative performance

Results are exported to `/results/`.

## Methodology

The Fama-French 5-Factor model explains equity returns through systematic risk factors. Each stock's historical returns are regressed against these factors to estimate sensitivity coefficients (betas). These betas, combined with estimated factor premiums, generate forward-looking expected returns used for portfolio selection.

The backtesting engine simulates historical performance to assess the strategy's risk-adjusted return profile under realistic market conditions.

## Performance Metrics
```
Annualized Return: 25.63%
Sharpe Ratio: 1.08
Maximum Drawdown: -36.5%
```

### Prerequisites
- Python 3.11
- Required packages: yfinance, pandas, numpy, statsmodels, matplotlib

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python main.py
``` 

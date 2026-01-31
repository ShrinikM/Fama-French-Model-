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
Classical Factor Model: 
Calculates expected returns using factor loadings and historical factor premiums

```
ER_i = β_iM · RP_M + β_iSMB · RP_SMB + β_iHML · RP_HML + β_iRMW · RP_RMW + β_iCMA · RP_CMA
```
Machine Learning Enhancements:
Principal Component Analysis (PCA):
- Reduces multicollinearity among factor returns
- Extracts latent, orthogonal risk factors
- Used as inputs to downstream models

Random Forest Regression:
- Predicts future stock returns using factors, rolling betas, momentum, and volatility
- Automatically models non-linearities and feature interactions
- Provides feature importance for interpretability

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

The project is grounded in the Fama–French 5-Factor asset pricing framework, where equity returns are explained through systematic risk exposures. Traditional linear factor models provide a baseline for expected return estimation.

To improve predictive performance, PCA-derived latent factors and Random Forest regression are incorporated, allowing the model to capture non-linear dynamics and evolving relationships in financial data. All models are evaluated using out-of-sample backtests to ensure robustness and avoid look-ahead bias.

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

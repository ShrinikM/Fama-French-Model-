# 📈 Multi-Factor Asset Pricing Model & Portfolio Backtester  
### *A Complete Quantitative Research Pipeline Implemented in Python*

This project implements a **full end-to-end multi-factor investing model** based on the **Fama–French 5-Factor framework**, combined with a custom backtesting engine.  
It automates the entire workflow used in real quantitative research environments:

- Data ingestion (stock prices + Fama–French factors)  
- Data cleaning & preprocessing  
- Factor regression (OLS)  
- Beta estimation and rolling exposures  
- Expected return modeling  
- Portfolio construction  
- Backtesting & performance analytics  

The pipeline is fully automated through `main.py`, making it production-ready and easy to extend.

---

## 🚀 Key Features

### **✔ Automated Data Pipeline**
- Downloads historical prices for 50+ equities using *yfinance*  
- Loads and cleans Fama–French 5-factor data  
- Converts returns to proper formats  
- Merges factor data with stock returns seamlessly  

### **✔ Factor Regression (OLS)**
Uses `statsmodels` to compute each stock’s exposure to:

- **Market**
- **SMB** (Size factor)
- **HML** (Value factor)
- **RMW** (Profitability)
- **CMA** (Investment)

Outputs clean beta estimates for each stock.

### **✔ Expected Return Model**
Expected return for each stock is computed as:

\[
ER_i = \beta_{iM} \cdot RP_M +
       \beta_{iSMB} \cdot RP_{SMB} +
       \beta_{iHML} \cdot RP_{HML} +
       \beta_{iRMW} \cdot RP_{RMW} +
       \beta_{iCMA} \cdot RP_{CMA}
\]

Factor premiums are estimated as historical averages.

### **✔ Portfolio Construction**
- Selects **top 20 stocks** by expected return  
- Builds an **equal-weighted, long-only portfolio**  
- Can be extended to long/short, risk parity, or optimizer-based portfolios  

### **✔ Backtesting Engine**
Evaluates portfolio performance using:

- Daily returns  
- Cumulative performance  
- Annualized return  
- Sharpe ratio  
- Maximum drawdown  

Outputs results into `/results/`.

---

## 📊 Example Backtest Results
Annualized Return: 25.63%
Sharpe Ratio: 1.08
Max Drawdown: -36.5%

---

## 🧠 Methodology Overview

### **1. Factor Model (Fama–French 5-Factor)**
The model explains stock returns using five systematic drivers:

- **Market Risk (β · Market premium)**  
- **Size (Small minus Big)**  
- **Value (High minus Low)**  
- **Profitability (Robust minus Weak)**  
- **Investment (Conservative minus Aggressive)**  

Each stock's return is regressed on these factors using OLS.

### **2. Beta Estimation**
Betas represent sensitivities:

- Example: *NVDA βₘ = 1.8*  
  → If the market rises 1%, NVDA historically rises ~1.8%.

### **3. Expected Returns**
Stocks with high positive exposures to strong factor premiums receive higher expected returns.

### **4. Portfolio Construction**
The highest-expected-return stocks form a long-only portfolio.

### **5. Backtesting**
Simulates historical performance to evaluate the strategy’s risk/return profile.

---

## 🧩 Technologies Used

- **Python 3.11**  
- **Pandas**, **NumPy**  
- **statsmodels** (OLS regressions)  
- **yfinance** (data ingestion)  
- **Matplotlib** (for visualization in notebooks)  

---

## 🚧 Future Improvements (Roadmap) (Chat-GPT)

- Add **Streamlit dashboard** for interactive visualization  
- Add **pytest unit tests** for model validation  
- Add **rolling factor exposures** & time-varying betas  
- Add **machine learning factor models** (Random Forest, XGBoost, Lasso)  
- Integrate **risk model** (covariance shrinkage, Ledoit–Wolf)  
- Add **optimizer** (mean-variance, Black–Litterman, risk parity)  


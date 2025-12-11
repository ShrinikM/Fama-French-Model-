import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False
    )

    if "Adj Close" in df.columns:
        price_data = df["Adj Close"]
    else:
        price_data = df["Close"]

    return price_data



def load_fama_french(path):
    df = pd.read_csv(path)

    cleaned_columns = []
    for c in df.columns:
        cleaned_name = c.strip().replace(" ", "_")
        cleaned_columns.append(cleaned_name)
    df.columns = cleaned_columns

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.set_index("date", inplace=True)

    rename_map = {
        "Mkt-RF": "Market",
        "SMB": "SMB",
        "HML": "HML",
        "RMW": "RMW",
        "CMA": "CMA"
    }
    df = df.rename(columns=rename_map)
    return df




def merge_stock_factors(stock_returns, factors):
    merged = stock_returns.join(factors, how="inner")

    return merged

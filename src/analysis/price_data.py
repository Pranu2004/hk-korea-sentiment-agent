
import yfinance as yf
import pandas as pd
from pathlib import Path


STOCKS = {
    "Hong Kong": {
        "0700.HK": "Tencent",
        "9988.HK": "Alibaba",
        "3690.HK": "Meituan",
        "1810.HK": "Xiaomi",
        "0941.HK": "China Mobile"
    },
    "South Korea": {
        "005930.KS": "Samsung Electronics",
        "000660.KS": "SK Hynix",
        "005380.KS": "Hyundai Motor",
        "373220.KS": "LG Energy Solution",
        "035420.KS": "NAVER"
    }
}


def download_stock_data(ticker, period="2y"):
    try:
        stock = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if stock.empty:
            print(f"No data found: {ticker}")
            return pd.DataFrame()

        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        stock = stock.reset_index()
        stock["Ticker"] = ticker

        return stock

    except Exception as error:
        print(f"Error downloading {ticker}: {error}")
        return pd.DataFrame()


def download_all_stocks():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for exchange, companies in STOCKS.items():
        print(f"\nDownloading {exchange} stocks...")

        for ticker, company in companies.items():
            print(f"Downloading {company} ({ticker})...")

            data = download_stock_data(ticker)

            if not data.empty:
                data["Company"] = company
                data["Exchange"] = exchange

                all_data.append(data)

                filename = output_dir / f"{ticker.replace('.', '_')}.csv"
                data.to_csv(filename, index=False)

                print(f"Saved {len(data)} records.")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(output_dir / "all_stock_prices.csv", index=False)

        print("\nDOWNLOAD SUMMARY")
        print("=" * 45)
        print("Companies downloaded:", combined["Ticker"].nunique())
        print("Total price records:", len(combined))
        print("Saved in:", output_dir.resolve())
    else:
        print("No stock data was downloaded.")


if __name__ == "__main__":
    download_all_stocks()

import yfinance as yf

def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="5y")  # Fetch 5 years of data
    print(f"Fetched data for {ticker}: {hist.shape}")
    return hist

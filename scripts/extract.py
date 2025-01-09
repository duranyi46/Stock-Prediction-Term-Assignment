import yfinance as yf

def extract_data(index_list):
    """
    In this function, you should receive data with yfinance library.
    This list will be your raw data and you will use it in initial load to PostgreSQL and for daily load to PostgreSQL.

    Example
    -------
    :input: ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'ADBE', 'TSLA', 'FFIE', 'ASTI', 'ALLR']

    :output:

    list_finance : {}

    :return: 
    """
    list_finance = {}
    for ticker in index_list:
        try:
            print(f"Fetching data for {ticker}...")
            stock_data = yf.download(ticker, start="2010-01-01", end="2024-12-31")
            if not stock_data.empty:
                list_finance[ticker] = stock_data
                print(f"Data for {ticker} fetched successfully.")
            else:
                print(f"No data found for {ticker}.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return list_finance


import yfinance as yf
import pandas as pd

class StockScraper:
    def __init__(self, tickers, period='1y', interval='1d'):
        """
        Initialize scraper object, set the tickers you want to scraper

        Args:
            tickers : a list of stock tickers you wish to scrape
            period : timeframe in which you want data (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max). Default is '1y'
            interval : frequency of scraped data (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). Default is '1d'

        Returns:
            None
        """


        self.tickers = tickers
        self.interval = interval
        self.period = period


    def download_stock_data(self):
        df_list = list()

        for ticker in self.tickers:
            data = yf.download(ticker, period = self.period,
                interval = self.interval,
                group_by = 'ticker',
                threads = True
            )
            data['Ticker'] = ticker  # add this column because the dataframe doesn't contain a column with the ticker
            df_list.append(data)

        # combine all dataframes into a single dataframe
        df = pd.concat(df_list)

        #reorder so ticker on the left
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        # save to csv
        df.to_csv('ticker.csv')



if __name__ == "__main__":
    sc = StockScraper(["MSFT", "AAPL"])
    sc.download_stock_data()
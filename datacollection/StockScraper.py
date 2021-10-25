import yfinance as yf
import pandas as pd

class StockScraper:
    def __init__(self, tickers, period=None, startdate=None, enddate=None, interval='1d'):
        """
        Initialize scraper object, set the tickers you want to scraper

        Args:
            tickers : a list of stock tickers you wish to scrape
            period : timeframe in which you want data (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max).
            startdate : start date of data
            enddate : end date of data
            interval : frequency of scraped data (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). Default is '1d'

        NOTE: If period is blank, you must specify a start and end dates

        Returns:
            None
        """
        if period == None and startdate == None and enddate == None:
            raise Exception("You must specify one timeframe in order to scrape")
        
        if period == None and (startdate == None or enddate == None):
            raise Exception("You must specify both ends of the timeframe in order to scrape")

        if period not in ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']:
            raise Exception("Please input a valid period")
        
        if interval not in ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']:
            raise Exception("Please input a valid time interval")

        


        self.tickers = tickers
        self.interval = interval
        if period == None:
            self.startdate = startdate
            self.enddate == enddate
            self.isPeriod = False
        else:
            self.period = period
            self.isPeriod = True
        
        


    def download_stock_data(self):
        df_list = list()

        for ticker in self.tickers:
            if self.isPeriod:
                data = yf.download(
                    ticker, 
                    period = self.period,
                    interval = self.interval,
                    group_by = 'ticker',
                    threads = True
                )
            else:
                data = yf.download(
                    ticker, 
                    start = self.startdate, 
                    end = self.enddate,
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
    sc = StockScraper(["MSFT", "AAPL"], period='10y')
    sc.download_stock_data()
import pandas as pd
import datetime as dt
import numpy as np


class Data:
    start_date = dt.datetime(2001, 1, 1)
    end_date = dt.datetime(2021, 12, 30)
    """
    1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    duration = "1d"

    def __init__(self, ticker, ds_path):
        """

        :param ticker:
        :param ds_path: Dataset path
        """
        self._ticker = ticker
        # self._dataset_path_jse = 'Data/Dataset/JSE/Dataset.csv'
        self._dataset_path_jse = ds_path

    def get_price(self):
        """

        :return: reformats Dataframe converting Ticker column into 'adjclose'
        """
        prices = pd.read_csv(self._dataset_path_jse)
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices.set_index('date')
        prices = prices.rename(columns={self._ticker: 'adjclose'})
        prices['symbol'] = self._ticker
        prices = prices[['symbol', 'adjclose']].dropna()
        prices['adjclose'] = np.concatenate([prices['adjclose'].divide(100)])
        return prices

    def resample_data(self):
        """

        :return: this resamples the DataFrame dates into BusinessQuaters, and returns Adjclose
        """
        prices = self.get_price()
        # prices = prices.resample('M', convention='end').mean()
        return prices

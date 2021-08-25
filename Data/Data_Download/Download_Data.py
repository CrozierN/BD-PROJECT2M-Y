import pandas as pd
import requests
import json
import os
import datetime as dt
from datetime import date
import yahooquery as query


class Download_Data:
    def __init__(self, ticker_path):
        self._start_date = dt.datetime(2001, 1, 1)
        self._end_date = dt.datetime(2021, 12, 30)
        self._today = date.today()
        # self._ticker_path = 'Data/jse_tickers.csv'
        self._ticker_path = ticker_path
        self._dframe = pd.read_csv(self._ticker_path)

    def stock_tickers(self):
        return list(self._dframe['Ticker'])

    def company_sectors(self):
        return list(self._dframe.Sector.drop_duplicates().dropna())

    def get_prices(self, ticker_name):
        prices = pd.DataFrame()
        try:
            ticker = query.Ticker(ticker_name)
            prices = ticker.history(interval='1d', start=self._start_date, end=self._end_date)[['adjclose']]
            prices = (prices.dropna()).reset_index(level=['symbol', 'date'])
            prices['date'] = pd.to_datetime(prices['date'])
            prices['date'] = prices['date'] - pd.Timedelta(days=1)
            prices = prices.drop(columns=['symbol'])
            prices = prices.rename(columns={'adjclose': ticker_name})
            prices = prices.set_index(['date'])
            prices = prices[1:]
        except Exception as ex:
            print(f"{ticker_name} Skipped {self._today}")

        return prices

    def merge_dataframes(self):

        base = self.get_prices(self.stock_tickers()[0])

        # count = 0
        for ticker in self.stock_tickers()[1:]:
            df = self.get_prices(ticker)
            base = pd.concat([base, df], axis=1)
            print(f"{ticker}: Saved")
            # if count == 7:
            #     break

            # count += 1

        return base

    def save_stock_prices(self, dataset_path):
        df = self.merge_dataframes()
        df.to_csv(dataset_path, mode='a')

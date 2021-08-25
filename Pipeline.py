import pandas as pd
from Data import Data as dta
from Data.Preprocessing import transformations as transforms
from Data.Preprocessing.cross_validation import cross_validation as cv
from Model.ARIMA_model import ARIMA_Model as stats_Model
from Plots.acf_pacf import correlation_plots as cp
from sklearn.metrics import mean_squared_error
from math import sqrt
import time


def model_mse(model_values):
    df = model_values
    error = mean_squared_error(df['y'], df['yhat'], squared=True)
    return f'Mean Squared Error: {error * 100}'


class Pipeline:
    def __init__(self, sector, ticker_path, ds_path):
        """

        :param sector: Sector - Can be an empty string
        :param ticker_path: Ticker Path
        :param ds_path: Dataset Path
        """
        self._sector = sector
        self._ticker_path = ticker_path  # 'Data/jse_tickers.csv'
        self._data = pd.read_csv(self._ticker_path)
        self._dataset_path = ds_path

    def tickers(self):

        _tickers = list((pd.DataFrame(self._data
                                      .drop(columns=['Sector'])))['Ticker'].drop_duplicates())

        # this will give use True if Sector is empty and False if the Sector has data
        skip_condition = self._data['Sector'].empty == False
        if not skip_condition:  # Sector if not empty
            _tickers = list((pd.DataFrame(self._data[self._data['Sector'] == self._sector]
                                          .drop(columns=['Sector'])))['Ticker'].drop_duplicates())
            return _tickers

        return _tickers

    def price_returns(self, ticker):
        stock_price = dta.Data(ticker, self._dataset_path).resample_data()[['adjclose']]
        stock_prices = transforms.transformations(stock_price).box_cox_trans()
        return stock_prices

    @staticmethod
    def model_corr_plots(prices):
        corr_plots = cp(prices.price_boxcox)
        return corr_plots.acf_pacf_plots()

    @staticmethod
    def cross_val(prices):
        cross_val = cv(data=prices.price_boxcox, test_split=.20)
        return cross_val

    # first we have to a list of qs and ps
    # second fit the model based on these values
    # third use the error matrix (MSE) to evaluate these the models and then select the best model

    @staticmethod
    def optimal_model(cross_validation):

        rmse, order = float("inf"), tuple

        start = time.process_time()
        for p in range(0, 4):
            for d in range(0, 3):
                for q in range(0, 4):
                    if p == 0 and q == 0:  # if its white noise, then skip the loop
                        continue  # figure out how to deal with random uncorrelated noise?

                    try:
                        model_order = (p, d, q)
                        prediction_model = stats_Model(cross_validation, model_order)
                        model_data = prediction_model.model().fillna(0)
                        # print(model_data)

                        current_rmse = sqrt(mean_squared_error(  # Calculate the [Root Mean Squared Error]
                            model_data['y'], model_data['yhat'], squared=True))

                        if current_rmse < rmse:
                            rmse = current_rmse
                            order = model_order
                    except Exception as ex:
                        # print(ex)
                        continue
            # pbar.update(n=1)
        end = time.process_time()
        print(f"Done: {end - start}")

        return order, rmse

    @staticmethod
    def trained_model(model_order, cross_validation):

        prediction = stats_Model(cross_validation, model_order)
        yhat_and_y = prediction.model().fillna(0)

        rmse = sqrt(mean_squared_error(yhat_and_y['y'], yhat_and_y['yhat']))

        return yhat_and_y, rmse

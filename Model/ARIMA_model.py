from statsmodels.tsa.arima.model import ARIMA
import pandas as pd


class ARIMA_Model:
    def __init__(self, cross_val, order):
        self.cross_val = cross_val
        self.order = order

    def model(self):
        cv = self.cross_val
        model_order = self.order

        predictions = []

        for train, test in cv.sliding_window(window=12, horizon=1):
            model = ARIMA(train, order=model_order)
            model_fit = model.fit()
            forecasts = model_fit.forecast()
            date_time = test.index
            predictions.append([date_time[0], forecasts[0], test[0]])

        return pd.DataFrame(predictions, columns=['date', 'yhat', 'y']).set_index('date')

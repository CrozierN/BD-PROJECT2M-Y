import matplotlib
import matplotlib.pyplot as plt
import time


class prediction_plot:
    def __init__(self, data, ticker):
        self._data = data
        self._ticker = ticker

    def plot(self):
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 6))
        plt.title(f'{self._ticker} Prediction Test plot')
        plt.plot(self._data['yhat'], color='red')
        plt.plot(self._data['y'], color='blue')
        plt.legend(['yhat', 'y'])

        # conf_interval = (0.1 * self._data['expected'].std()) / self._data['expected'].mean()

        # plt.fill_between(self._data['predicted'],
        #                 (self._data['expected'] - conf_interval),
        #                 (self._data['expected'] + conf_interval),
        #                 color='red',
        #                 alpha=0.1
        #                 )
        plt.show()
        plt.close()

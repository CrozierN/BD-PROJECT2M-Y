from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt


class correlation_plots:
    def __init__(self, data):
        self._data = data

    def acf_pacf_plots(self):
        df_box_price = self._data
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        # plt.title(df_box_price[''])
        lag_acf = plot_acf(df_box_price['price_boxcox'], zero=False, lags=20, ax=ax1)
        lag_pacf = plot_pacf(df_box_price['price_boxcox'], zero=False, lags=20, ax=ax2)
        f.tight_layout()
        plt.show()

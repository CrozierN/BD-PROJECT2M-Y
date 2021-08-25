from scipy import stats
import numpy as np


class transformations:

    def __init__(self, data):
        self.data = data

    def log_trans(self):
        df = self.data
        df['log_price'] = np.log(df['adjclose'])
        return df

    def log_ma(self):
        df = self.log_trans()
        df['log_price_moving_avg'] = df['log_price'].rolling(window=4,
                                                             center=False).mean()
        return df.dropna()

    def make_log_stationary(self):

        df = self.log_trans()
        df['log_price_diff'] = np.concatenate([df['log_price'].diff().dropna(), [0]])
        return df[['log_price_diff']]

    def box_cox_trans(self):
        df = self.log_trans()
        try:
            df['price_boxcox'] = stats.boxcox(df['adjclose'])[0]
            return df
        except Exception:
            df['price_boxcox'] = 0
        return df

    def make_box_stationary(self):
        df = self.box_cox_trans()
        df['box_price_diff'] = np.concatenate([df['price_boxcox'].diff().dropna(), [0]])
        return df

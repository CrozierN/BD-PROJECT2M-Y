import pandas as pd


class cross_validation:

    def __init__(self, data, test_split):
        """

        :param data: stock prices
        :param test_split: percentage (0.4)
        """
        self.data = data
        self.test_split = test_split

    def train_test_split(self):
        df = self.data
        y_split = round((len(df) * self.test_split))
        x = df[:round(len(df) - y_split)]
        y = df[-y_split:]
        return x, y

    def in_sample(self):
        x, _ = self.train_test_split()
        return x

    def out_of_sample(self):
        _, y = self.train_test_split()
        return pd.DataFrame(y)

    """
    We have to include a number of sample data
    """

    def sliding_window(self, window, horizon):
        train_data, _ = self.train_test_split()
        for i in range(len(train_data) - window - horizon + 1):
            split_train_data = train_data[i: window + i]
            split_valuation_data = train_data[i + window: window + i + horizon]
            yield split_train_data, split_valuation_data

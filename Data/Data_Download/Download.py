from Download_Data import Download_Data as DD


class Download:
    def __init__(self, ticker_path, save_path):
        # self.ticker_path = './Data/SA_ETF_tickers.csv'
        # self.save_path = 'Data/EFT/ETF_Historical_Data.csv'
        self.ticker_path = ticker_path
        self.save_path = save_path

    def save(self) -> None:

        data = DD(ticker_path=self.ticker_path)
        data.save_stock_prices(self.save_path)

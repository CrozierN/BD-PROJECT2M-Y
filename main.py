import Pipeline as pL
import warnings
from ast import literal_eval as make_tuple
import time
from Plots.prediction_plot import prediction_plot as pp
from Data.Data_Download.Download_Data import Download_Data as DD
import pandas as pd
import numpy as np
import datetime as dt
from Data.Data import Data as dta


def save_best_model(ticker_path, ds_path) -> None:
    dd = DD(ticker_path)  #

    comp_sectors = dd.company_sectors()

    # this will give us True if Sector is empty and False if the Sector has data
    skip_condition = pd.DataFrame(comp_sectors).empty == False
    sector = ''
    model_save_path = 'Model/SA_ETF_best_model.csv'
    if skip_condition:  # if sector is empty then skip code
        print(f"{comp_sectors}")
        sector = input("Choose Sector: ")
        model_save_path = './Model/best_models.csv'

    Pipeline = pL.Pipeline(sector, ticker_path, ds_path)
    tickers = Pipeline.tickers()
    print(tickers)

    ds = pd.read_csv(model_save_path).drop(columns=['Unnamed: 0'])  # Best Model Data

    for index in range(len(tickers)):
        ticker = tickers[index]
        skip_condition = (ds[ds['Ticker'] == ticker].any()).Sector

        if not skip_condition:  # if this is not true
            print(f"{ticker} Process Starting ....")
            prices = Pipeline.price_returns(ticker)
            cross_val = Pipeline.cross_val(prices)
            warnings.filterwarnings("ignore")
            try:
                (p, d, q), score = Pipeline.optimal_model(cross_val)

                print(f"Model Name: ARIMA({p}, {d}, {q}) -> {score}")

                model_name = f"ARIMA({p}, {d}, {q})"
                model_order = (p, d, q)
                print('\n Function: save_best_model')
                print(f'Order: {model_order}')

                best_model = pd.DataFrame([[model_name, sector, ticker, model_order, score]],
                                          columns=['Model_Name', 'Sector', 'Ticker', 'Model_Order', "Score"])
                print(f'Model Order')
                print(best_model)
                best_model.to_csv(model_save_path, mode='a', header=False)
            except Exception as ex:
                print(f'Ticker is skipped: {ticker} {ex}')
                skipped_ticker = pd.DataFrame([ticker], columns=['Ticker'])
                skipped_ticker.to_csv('skipped_tickers.csv', mode='a', header=False)
            print(f"{ticker} Process Ending ....")
        elif skip_condition:
            print(f"{ds[ds['Ticker'] == ticker].Ticker} Exists!!")


def predictions_(ticker_path, ds_path) -> None:
    saved_best_models = './Model/best_models.csv'
    # saved_best_models = 'Model/SA_ETF_best_model.csv'
    df = pd.read_csv(saved_best_models).drop(columns=['Unnamed: 0'])
    df['Model_Order'] = [make_tuple(tup) for tup in df.Model_Order]

    sectors = list(df.Sector.drop_duplicates())
    print(sectors)

    # this will give us True if Sector is empty and False if the Sector has data
    skip_condition = pd.DataFrame(sectors).empty == False
    sector = ''
    model_save_path = 'Model/SA_ETF_best_model.csv'
    if skip_condition:  # if sector is empty then skip code
        print(f"{sectors}")
        avail_sectors = input('Select an Available Sector: ')
        model_save_path = './Model/best_models.csv'

    answer = input('{y} To Continue and {n} to stop: ')

    for i in range(len(df)):
        if answer == 'y':
            data = df.iloc[i]

            Pipeline = pL.Pipeline(sector, ticker_path, ds_path)
            prices = Pipeline.price_returns(data.Ticker)
            cross_val = Pipeline.cross_val(prices)

            # plot_answer = input('Do you want to ACF and PACF? {y}: ')
            # if plot_answer == 'y':
            #     Pipeline.model_corr_plots(prices)
            #     time.sleep(1)

            warnings.filterwarnings("ignore")
            predictions, rmse = Pipeline.trained_model(data.Model_Order, cross_val)

            frame2 = Pipeline.cross_val(prices).out_of_sample().rename(
                columns={'price_boxcox': 'y'},
                inplace=False)

            predictions = pd.concat([predictions, frame2])

            print(f"{data.Ticker} -> Model: {data.Model_Name} -> RMSE: {data.Score}")
            print(f"{predictions} \n")

            plot_answer = input('Do you want to plot? {y}: ')
            if plot_answer == 'y':
                pp(predictions, data.Ticker).plot()
                time.sleep(1)

            # predictions['yhat'].to_csv('Data')

            answer = input('{y} to Continue and {n} to stop: ')

        elif answer == 'n':
            print('Terminated')
            break


if __name__ == '__main__':
    # path = 'Data/Dataset/EFT/SA_ETF_tickers.csv'
    # dataset_path = 'Data/Dataset/EFT/ETF_Historical_Data.csv'
    path = 'Data/Dataset/JSE/jse_tickers_sector.csv'
    dataset_path = 'Data/Dataset/JSE/Dataset.csv'
    # save_best_model(path, dataset_path)

    predictions_(path, dataset_path)

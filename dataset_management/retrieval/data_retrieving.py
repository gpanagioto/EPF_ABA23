import pandas as pd
import datetime as dt
import numpy as np
from entsoe import EntsoePandasClient
import os
import time
import yfinance as yf

def DataSave(data: pd.DataFrame, save_path: str, name: str) -> None:
    
    data.to_csv(save_path + name + '.csv')
    print(f"Size of {name} is {data.shape[0]} rows.\n")
    print(save_path)

def Directory(save_path, data_type, country):
    
    print(save_path)
    if not os.path.exists(save_path + '/' + data_type + '/' + country + '/' ):
        os.makedirs(save_path + '/' + data_type + '/' + country + '/' )
    data_save_path = save_path + '/' + data_type + '/' + country + '/' 
    print(data_save_path)
    
    return data_save_path

class EntsoeDataRetrieval():

    def __init__(self, client: EntsoePandasClient, start_date: str, end_date:str, country_code:str) -> None:
    
        self.client = client
        self.start_date = start_date
        self.end_date = end_date
        self.country = country_code
    
    def DayAheadPrices(self, save_path: str) -> None:
        day_ahead_prices = self.client.query_day_ahead_prices(self.country, start = self.start_date, end = self.end_date)
        DataSave(day_ahead_prices, save_path, 'Day_Ahead_Prices')


    def LoadAndForecast(self, save_path: str) -> None:

        load_and_forecast = self.client.query_load_and_forecast(self.country, start = self.start_date, end = self.end_date)
        DataSave(load_and_forecast, save_path, 'Load_And_Forecast')

    def WindSolarForecast(self, save_path: str) -> None:

        wind_solar_forecast = self.client.query_wind_and_solar_forecast(self.country, start = self.start_date, end = self.end_date)
        DataSave(wind_solar_forecast, save_path, 'Wind_Solar_Forecast')

    
    def Imports(self, save_path: str) -> None:

        imports = self.client.query_import(country_code = self.country, start = self.start_date, end = self.end_date)
        DataSave(imports, save_path, 'Imports')


    def Exports(self, save_path: str, country_to:str) -> None:

        name = country_to + '_Exports'
        exports = self.client.query_crossborder_flows(self.country, country_code_to=country_to, start = self.start_date, end = self.end_date)
        DataSave(exports,  save_path, name)


def YahooDataRetrieval(save_path: str, ticker: str, start_date, end_date) -> None:
        
    name = ticker

    results = yf.Ticker(ticker)

    yahoo_data = yf.download(ticker, start_date, end_date)

    DataSave(yahoo_data, save_path, name)

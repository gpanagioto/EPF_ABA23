import pandas as pd
import datetime as dt
import numpy as np
from entsoe import EntsoePandasClient
import os

def DataSave(data: pd.DataFrame, save_path: str, name: str) -> None:
    
    data.to_csv(save_path + name + '.csv')
    print(f"Size of {name} is {data.shape[0]} rows.\n")

def Directory(data_type, country, country_to):
     
    if data_type != "exports":
        if not os.path.exists('data' + '/' + data_type + '/' + country): 
            os.makedirs('data' + '/' + data_type + '/' + country)
        save_path = 'data' + '/' + data_type + '/' + country + '/'
    else:
        if not os.path.exists('data' + '/' + data_type + '/' + country + '/' ):
            os.makedirs('data' + '/' + data_type + '/' + country + '/' )
        save_path = 'data' + '/' + data_type + '/' + country + '/' 

    return save_path

class DataRetrieval():

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
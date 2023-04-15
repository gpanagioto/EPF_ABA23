import pandas as pd
import datetime as dt
import numpy as np
from entsoe import EntsoePandasClient


def DataSave(data: pd.DataFrame, save_path: str, name: str) -> None:
    
    data.to_csv(save_path + name + '.csv')
    print(f"Size of {name} is {data.shape[0]} rows.\n")

class DataRetrieval():

    def __init__(self, save_path: str, client: EntsoePandasClient, start_date: str, end_date:str, country_code:str) -> None:
    
        self.save_path = save_path
        self.client = client
        self.start_date = start_date
        self.end_date = end_date
        self.country = country_code

    def Main(self) -> None:

        
        #
        day_ahead_prices = self.client.query_day_ahead_prices(self.country, start = self.start_date, end = self.end_date)
        DataSave(day_ahead_prices, self.save_path, 'Day_Ahead_Prices')

        #
        load_and_forecast = self.client.query_load_and_forecast(self.country, start = self.start_date, end = self.end_date)
        DataSave(load_and_forecast, self.save_path, 'Load_And_Forecast')

        #
        wind_solar_forecast = self.client.query_wind_and_solar_forecast(self.country, start = self.start_date, end = self.end_date)
        DataSave(wind_solar_forecast, self.save_path, 'Wind_Solar_Forecast')

    
    def Imports(self) -> None:
       
        imports = self.client.query_import(country_code = self.country, start = self.start_date, end = self.end_date)
        DataSave(imports, self.save_path, 'Imports')


    def Exports(self, country_to:str) -> None:

        exports = self.client.query_crossborder_flows(self.country, country_code_to=country_to, start = self.start_date, end = self.end_date)
        DataSave(exports, self.save_path, 'Exports')
import pandas as pd
import datetime as dt
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pytz

files_names = {
               "imports":"Imports.csv",
               "DE_LU":"DE_LU_Exports.csv",
               "DK_1": "DK_1_Exports.csv",
               "SE_4": "SE_4_Exports.csv",
               "Day_Ahead_Prices":"Day_Ahead_Prices.csv",
               "Load_And_Forecast":"Load_And_Forecast.csv",
               "Wind_Solar_Forecast":"Wind_Solar_Forecast.csv"
              }

def FindFile(file_name, search_path):
    """Search for a file with a specific name in the given search path."""
    for dir_path, _, file_list in os.walk(search_path):
        # Look for the file in the current directory
        if file_name in file_list:
            return os.path.join(dir_path, file_name)

    # If the file was not found
    return None


class Merging():

    def __init__(self, path) -> None:
        
        self.search_path = sys.path[0]

    def Timestamp(self, file_name):

        file = FindFile(file_name, self.search_path)
        
        df = pd.read_csv(file, index_col = 0)
        df.index.rename('Timestamp', inplace=True)
        df.rename(columns={'0': 'Electricity'})

        if df.columns[0] == 'Unnamed: 0' and '0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        try:
            if file_name == "Day_Ahead_Prices.csv":
                name = 'Electricity Price'
            else:
                name = 'Electricity Quantity'

            df.rename(columns={'0':name}, inplace = True, errors="raise")
            
        except:
            pass

        df.index = pd.to_datetime(df.index, format = '%Y %m %d %H:%M:%S', utc = False)        
        df.index = df.index.to_series().apply(lambda row: row.astimezone(pytz.timezone("Europe/Copenhagen")))   
        df.index = df.index.to_series().apply(lambda row: row.replace(tzinfo=None))
        
        return df

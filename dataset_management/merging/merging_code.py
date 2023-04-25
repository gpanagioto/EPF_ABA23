import pandas as pd
import datetime as dt
import os
import functools as ft
import inspect
import datetime as datetime

def find_file(file_name: str, search_path: str) -> str: 
    """Search for a file with a specific name in the given search path."""
    for dir_path, _, file_list in os.walk(search_path):
        # Look for the file in the current directory
        if file_name in file_list:
            return os.path.join(dir_path, file_name)

    # If the file was not found
    return None

def find_folder(folder_name: str, search_path: str) -> str:
    for root, dirs, files in os.walk(search_path):
        if folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            return folder_path
    
    #if the folder not found
    return None

def get_variable_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]

class DataMerging():

    def __init__(self, path: str) -> None:
        
        self.search_path = path

    def entsoe_timestamp(self, file_name: str) -> pd.DataFrame:
        
        file = find_file(file_name, self.search_path)
        
        df = pd.read_csv(file, index_col = 0)
        df.index.rename('Timestamp', inplace=True)

        if df.columns[0] == 'Unnamed: 0' and '0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        try:
            if file_name == "Day_Ahead_Prices.csv":
                name = file_name.split('.')[0]
            else:
                name = file_name.split('.')[0] + '_Volume'

            df.rename(columns={'0':name}, inplace = True, errors="raise")
            
        except:
            pass

        df.index = pd.to_datetime(df.index, format = '%Y %m %d %H:%M:%S', utc = False)
        df.index = df.index.to_series().apply(lambda row: (row - dt.timedelta(hours=1)) if str(row).split('+')[1]=='02:00' else row)        #df.index = df.index.to_series().apply(lambda row: row.astimezone(pytz.timezone("Europe/Copenhagen")))   
        df.index = df.index.to_series().apply(lambda row: row.replace(tzinfo=None))
        
        return df.groupby(pd.Grouper(freq='H')).sum().sort_index()

    def yahoo_timestamp(self, file_name: str) -> pd.DataFrame:

        file = find_file(file_name, self.search_path)

        try:
            df = pd.read_csv(file, index_col=0, header=0, usecols=['Date','Price'], parse_dates=['Date'], infer_datetime_format=True)
            df.rename(columns={'Price': file_name.split('.')[0] + '_Price'}, inplace=True)

        except:
            df = pd.read_csv(file, index_col=0, header=0, usecols=['Date','Close'], parse_dates=['Date'], infer_datetime_format=True)
            df.rename(columns={'Close': file_name.split('.')[0] + '_Price'}, inplace=True)

        df.index.rename('Timestamp', inplace=True)
        

        return df.sort_index()
    
    def energy_data_timestamp(self, file_name: str, columns: list) -> pd.DataFrame:

        file = find_file(file_name, self.search_path)
        if columns:
            usecols = columns
        else:
            usecols = [ 'MTU', 'Biomass  - Actual Aggregated [MW]',
                        'Waste  - Actual Aggregated [MW]']
            
        try:
            df = pd.read_csv(file, usecols=usecols)
           
            df['Timestamp_cet'] = df['MTU'].apply(lambda row: row.split('-')[0])
            df['Timestamp'] = df['Timestamp_cet'].apply(lambda row: datetime.datetime.strptime(row, '%d.%m.%Y %H:%M '))
            df.drop(['Timestamp_cet','MTU'], inplace=True, axis=1)
            df.set_index('Timestamp', inplace=True)

        except Exception as e: 
            print(e)
            pass

        return df.sort_index()


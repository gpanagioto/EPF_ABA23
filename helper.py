import pandas as pd
import holidays
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

# Prophet model for time series forecast
# !pip install yfinance prophet

import warnings
warnings.filterwarnings('ignore')

# add year, quarter, month, day, date and hour columns
def get_dt_info(df:pd.DataFrame(), dt_col:str,
                yr:bool = False, qt:bool = False, mo:bool = False,
                day:bool = False, date:bool = False, w:bool = False, h:bool = False) -> pd.DataFrame():
    if yr:
        df['Year'] = df[dt_col].dt.year
    if qt:
        df['Quarter'] = df[dt_col].dt.quarter
    if mo:
        df['Month'] = df[dt_col].dt.month
    if day:
        df['Day'] = df[dt_col].dt.day
    if date:
        df['Date'] = df[dt_col].dt.date
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y %m %d')
    if w:
        if df[dt_col].dtype == '<M8[ns]':
            df['Week'] = df[dt_col].dt.isocalendar().week
        else:
            print(dt_col,'is not a date!')
    if h:
        df['Hour'] = df[dt_col].dt.hour
        
    return df

def add_business_days(df:pd.DataFrame(), country:str, inplace:bool = False) -> pd.DataFrame():
    country_calendar = getattr(holidays, country)
    custom_business_days = pd.tseries.offsets.CustomBusinessDay(calendar = country_calendar)
    start = min(df['Date']) # generalitzar columna?
    end = max(df['Date']) # generalitzar columna?
    print('From', start, 'to', end)
    business_days = pd.bdate_range(start, end, freq = custom_business_days)
    if inplace:
        df['Business'] = df['Date'].isin(business_days) # generalitzar columna?
        return df
    else:
        df2 = df.copy()
        df2['Business'] = df2['Date'].isin(business_days) # generalitzar columna?
        return df2
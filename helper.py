import pandas as pd
import numpy as np
import datetime as dt

def remove_utc(col_name, # str: name of the column that contains the object to convert to timestamp
               tz_offset, # int: timezone offset. E.g., CET = +1
               df # dataframe: contains all the info
              ):
    df['Timestamp'] = pd.to_datetime(df[col_name],utc = True)#, format = '%Y-%m-%d %H:%M:%S')
    df['Timestamp'] = (df['Timestamp'] + dt.timedelta(hours = tz_offset)).dt.tz_localize(None)
    df.drop([col_name], axis = 1, inplace = True) # drop the column
    df.set_index('Timestamp', inplace = True) # set column 'Timestamp' as index
    return df
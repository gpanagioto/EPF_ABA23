import pandas as pd
import datetime as dt
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def FindFile(file_name, search_path):
    """Search for a file with a specific name in the given search path."""
    for dir_path, _, file_list in os.walk(search_path):
        # Look for the file in the current directory
        if file_name in file_list:
            return os.path.join(dir_path, file_name)

    # If the file was not found
    return None

def Timestamp(col_name, tz_offset, df):

    df['Timestamp'] = pd.to_datetime(df[col_name], format = '%Y %m %d %H:%M:%S',utc = True)

    df['Timestamp'] = (df['Timestamp'] + dt.timedelta(hours = tz_offset)).dt.tz_localize(None)

    df.drop([col_name], axis = 1, inplace = True) # drop the column

    df.set_index('Timestamp', inplace = True) # set column 'Timestamp' as index

    return df
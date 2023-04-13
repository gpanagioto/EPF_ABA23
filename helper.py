import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import datetime as dt

def split_timeseries(df, train_s, train_e, test_s, test_e): # format of dates has to be YYYY-mm-dd
    if (train_e <= train_s):
        print('Incoherent dates for the TRAIN set!')
    elif (test_e <= test_s):
        print('Incoherent dates for the TEST set!')
    else:
        X_train = df.loc[train_s:train_e]
        X_test = df.loc[test_s:test_e]
        
        return X_train, X_test
    
def split_timeseries1(df, train_start, cnt, method, perc = 0.85): # train_start array of dates; formatted YYYY-mm-dd
    
    if cnt == len(train_start)-1:
        end=df['Date'].max()
    else:
        end = train_start[cnt +1] - dt.timedelta(days=1)

    if method == 0:
        start = train_start[cnt]
        delta=(end-start).days
        train_end = start + pd.DateOffset(days=round(perc*delta,0))
        test_start = train_end + pd.DateOffset(days=1)

    elif method == 1:
        start = train_start[0]
        delta=(end-start).days
        train_end = end - pd.DateOffset(days=91)
        test_start = train_end + pd.DateOffset(days=1)
    

    X_train = df.loc[start:train_end]
    X_test = df.loc[test_start:end]
       
    return X_train, X_test
    
def standardize(train_set, test_set, cols):
    mu = train_set[cols].mean()
    std = train_set[cols].std()
    
    train_set_std = (train_set - mu) / std
    test_set_std = (test_set - mu) / std
    
    return train_set_std, test_set_std
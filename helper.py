import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
import datetime as dt

def read_our_data(file_name):
    file_dir = './data/'+file_name
    df = pd.read_csv(file_dir)
    cols_check = ['Timestamp', 'Date']
    for col in cols_check:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

# def split_timeseries(df, train_s, train_e, test_s, test_e): # format of dates has to be YYYY-mm-dd
#     if (train_e <= train_s):
#         print('Incoherent dates for the TRAIN set!')
#     elif (test_e <= test_s):
#         print('Incoherent dates for the TEST set!')
#     else:
#         X_train = df.loc[train_s:train_e]
#         X_test = df.loc[test_s:test_e]
        
#         return X_train, X_test

def lag_df(df, col, lags):
    df_lagged = df.copy()
    for lag in lags:
        new_col = str(col) + '-lag' + str(lag)
        df_lagged[new_col] = df[col].shift(lag)
    return df_lagged

def split_timeseries(df, train_start, cnt, method, perc = 0.85): # train_start array of dates; formatted YYYY-mm-dd
    # method {0 - moving blocks, 1 - from the begging}
    if 'Timestamp' in df.columns:
        df.set_index('Timestamp', inplace = True)

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

def get_feature_target(df, features, target):
    df.reset_index(inplace = True)
    if 'Timestamp' in df.columns:
        df.drop(['Timestamp'], axis = 1)
    X = df[features]
    y = df[target]

    return X, y

def standardize(train_set, test_set, cols):
    mu = train_set[cols].mean(axis = 0)
    std = train_set[cols].std(axis = 0)
    
    train_set_std = train_set.copy()
    test_set_std = test_set.copy()
    train_set_std[cols] = (train_set[cols] - mu) / std
    test_set_std[cols] = (test_set[cols] - mu) / std
    
    return train_set_std, test_set_std

def model_evaluation(true, pred):
    
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(true, pred))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(true, pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(true, pred)))
    mape = np.mean(np.abs((true - pred) / np.abs(true)))
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*(1 - mape), 2))
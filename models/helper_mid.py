import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor 
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from numpy import savetxt

import datetime as dt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers
from time_analysis import *

import xgboost as xg

def drop_lag_df(df:pd.DataFrame(), new_col:str, lags:list) -> pd.DataFrame():
    df_lagged = df.copy()
    for lag in lags:
        col = str(new_col) + '-lag' + str(lag)
        print(col, df_lagged[df_lagged[col].isna()]['Date'].unique())
        idx_drop=(df_lagged[df_lagged[col].isna()]['Date'].unique())
        df_lagged.drop(df_lagged[df_lagged['Date'].isin(idx_drop)].index, inplace = True)
    return df_lagged


def split_timeseries(df:pd.DataFrame(), train_start:list, cnt:int, method:int, perc = 0.85) -> pd.DataFrame(): # train_start list of dates; formatted YYYY-mm-dd
    # method {0 - moving blocks, 1 - from the begging}
    df.head()
    if 'Timestamp' in df.columns:
        df.set_index('Timestamp', inplace = True)

    if cnt == len(train_start)-1:
        end=pd.to_datetime(df['Date'].max()) + dt.timedelta(hours = 23)

    else:
        end = train_start[cnt +1] - dt.timedelta(hours = 1)
        the_end = train_start[cnt +1] - dt.timedelta(hours = 2)

    if method == 0:
        start = train_start[cnt]
        delta=(end-start).days
        train_end = start + pd.DateOffset(days=round(perc*delta,0), hours = 23)
        test_start = train_end + pd.DateOffset(hours = 1)

    elif method == 1:
        start = train_start[0]
        delta=(end-start).days
        train_end = end - pd.DateOffset(days = 30)
        test_start = train_end + pd.DateOffset(hours = 1)

    print(f'train {start} - {train_end}, test {test_start} - {end}')
    trainset = df.loc[start:train_end]
    testset = df.loc[test_start:end]
       
    return trainset, testset

def get_feature_target(df, features, target):
    df.reset_index(inplace = True)
    if 'Timestamp' in df.columns:
        df.drop(['Timestamp'], axis = 1)
    X = df[features]
    y = np.array(df[target]).ravel()

    return X, y

def standardize(train_set, test_set, cols):
    mu = train_set[cols].mean(axis = 0)
    std = train_set[cols].std(axis = 0)
    
    train_set_std = train_set.copy()
    test_set_std = test_set.copy()
    train_set_std[cols] = (train_set[cols] - mu) / std
    test_set_std[cols] = (test_set[cols] - mu) / std
    
    return train_set_std, test_set_std

#%%
def build_lr(x_train, y_train, x_test): # CONSIDER USING STATSTOOLS.OLS AS IT PROVIDES TTEST
    model_ = LinearRegression()
    model_.fit(x_train, y_train)

    ypred = model_.predict(x_test)
    params = model_.get_params()

    return ypred, params

def build_ma(df, target, pred_window): # pred_window has to be in hours
    start = min(df.index)
    end = max(df.index) + dt.timedelta(hours = pred_window)
    timeframe = pd.date_range(start = start, end = end, freq = '1H')

    df_ma = pd.DataFrame( index = timeframe)
    df_ma = df_ma.join(df[target])
    col_name = 'lag' + str(pred_window)
    df_ma[col_name] = df_ma[target].shift(pred_window)
    df_ma['pred'] = df_ma[col_name].rolling(5, closed = 'left').mean() # inmediate -1, -2, -3, -4, -5 lags were found relevant

    return df_ma

def build_rf(x_train, y_train, x_test, random_search = True):
    rf = RandomForestRegressor(random_state = 42, min_samples_split = 10)
    if random_search: # perform randomized search cross validation
        rf_grid = {'max_depth': [3, 7, 10],
                   'min_samples_split': np.arange(2, 30, 2),
                   'min_samples_leaf': np.arange(2, 15, 2)}
        model_ = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 30, cv = 3, random_state = 42, n_jobs = -1)
        model_.fit(x_train, y_train)
        model_.best_estimator_.get_params() # optimal parameters
        model_ = model_.best_estimator_ # evaluate optimal model
    else:
        model_.fit(x_train, y_train)
    ypred = model_.predict(x_test)
    params = model_.get_params()
    
    return ypred, params

def build_xgb(x_train, y_train, x_test):
    x_train.columns = x_train.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))
    x_test.columns = x_test.columns.str.translate("".maketrans({"[":"{", "]":"}","<":"^"}))
    model_ = xg.XGBRegressor(seed = 42)
    model_.fit(x_train, y_train)
    ypred = model_.predict(x_test)
    return ypred, model_

def build_gb(x_train, y_train, x_test):
    model_ = GradientBoostingRegressor(random_state = 42)
    model_.fit(x_train, y_train)
    
    ypred = model_.predict(x_test)

    return ypred, model_

def build_far(x_train, y_train, x_test,steps):
    model_ = ForecasterAutoreg(regressor = Ridge(),lags = 336)
    y_train=pd.Series(data=y_train)
    model_.fit(y_train,exog=x_train)
    print(x_train.index[0])
    print(x_test.index[0])
    ypred = model_.predict(steps,exog=x_test)
    return ypred, model_

def fit_moving_average(timeseries, lag=3):
    return list([np.average(timeseries.iloc[max(i-lags,0):i-1]) for i in range(1,len(timeseries)+1)])

def build_lstm (x_dev, y_dev, x_test):

    tf.random.set_seed(42)

    x_train, x_val = train_test_split(x_dev,test_size = 0.2, shuffle=False)
    y_train, y_val = train_test_split(y_dev,test_size = 0.2, shuffle=False)

    # create and fit the LSTM network
    model_ = keras.models.Sequential()
    model_.add(layers.LSTM(64, input_shape = (x_train.shape[1], 1)))
    model_.add(layers.Dense(32))
    model_.add(layers.Dense(1))
    model_.compile(loss='mean_squared_error', optimizer='adam')
    # change data type so it can be converted to a tensor
    x_train = np.asarray(x_train).astype(np.float32)
    x_val = np.asarray(x_val).astype(np.float32)
    x_test = np.asarray(x_test).astype(np.float32)
    model_.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data = (x_val, y_val))

    ypred = model_.predict(x_test)
    print(ypred.shape)

    return ypred, model_

def run_model(model_type, df, k_folds, split_method, train_start, features, target, cols_std, X_std = True): # argument timeframe is necessary to choose baseline
    # model_type: lr (linear regression), rf (random forest), xgb (XGBoost), lstm (long-short term memory - recursive neural network)

    ypred = []
    models = []

    for k in range(k_folds):
        print('Iteration ', k)

        # split in train and test set
        train_set, test_set = split_timeseries(df, train_start, k, method = split_method)

        # get features and target
        X_train, y_train = get_feature_target(train_set, features, target)
        X_test, y_test = get_feature_target(test_set, features, target)

        # standardize
        if X_std == True:
            X_train, X_test = standardize(X_train, X_test, cols_std)
        if model_type == 'lr':
            yhat, model_ = build_lr(X_train, y_train, X_test)
        elif model_type == 'rf':
            yhat, model_ = build_rf(X_train, y_train, X_test, random_search = True)
        elif model_type == 'xgb':
            yhat, model_ = build_xgb(X_train, y_train, X_test)
        elif model_type == 'gb':
            yhat, model_ = build_gb(X_train, y_train, X_test)
        elif model_type == 'lstm':
            yhat, model_ = build_lstm(X_train, y_train, X_test)
            yhat = yhat.reshape(-1)
        
        model_evaluation(y_test, yhat)
        
        ypred.append(yhat)
        models.append(model_)
            
    return ypred, models


def run_time_model(model_type, df, k_folds, split_method, train_start, features, target,steps,exog=None,hourly=None): # argument timeframe is necessary to choose baseline
    # model_type: lr (linear regression), rf (random forest), xgb (XGBoost), lstm (long-short term memory - recursive neural network)

    ypred = []
    models = []

    for k in range(k_folds):
        print('Iteration ', k)

        # split in train and test set
        train_set, test_set = split_timeseries(df, train_start, k, method = split_method)
        train_set0 = train_set.copy()
        test_set0 = test_set.copy()
        train_set0 = train_set.copy()
        # get features and target
        X_train, y_train = get_feature_target(train_set0, features, target)
        X_test, y_test = get_feature_target(test_set0, features, target)


        if model_type == 'arima':
            if hourly==None:
                yhat = test_arima_(df,test_set,target,steps,hourly=False,season=False,exog=exog)
            else:
                yhat = test_arima_(df,test_set,target,steps,hourly=True,season=False,exog=exog)
            yhat = yhat[:steps]
            if exog!=None:
                yhat=yhat.values
        if model_type == 'far':
            yhat, model_ = build_far(X_train, y_train, X_test,steps)
            yhat=yhat.reset_index().copy()
            yhat=yhat['pred']
        #""" the code to obtain the values for the scheduler, one value per each quarter
        if k==0:
            print(test_set0.index)# to identify the index of the tested data
            a_repeated = np.repeat(yhat, 4)
            b_repeated = np.repeat(y_test[:steps], 4)
            a_repeated=np.around(a_repeated, decimals=2).astype(int)
            b_repeated=np.around(b_repeated, decimals=2).astype(int)
            savetxt('data_forecasted.csv',  a_repeated, delimiter='/n')
            savetxt('data.csv',  b_repeated, delimiter='/n')
        #"""
        
        plt.plot(y_test[:steps], label = 'True')
        plt.plot(yhat, label = 'Predicted')
        plt.title('True data vs prediction')
        plt.legend()
        plt.show()
        #model_evaluation(y_test[:steps], yhat)
        
        ypred.append(yhat)
            
    return ypred, models

def model_evaluation(true, pred):
    idx = np.where(true != 0) # avoid dividing by 0 in mape calculation
    true2 = true[idx]
    pred2 = pred[idx]
    plt.plot(true2, label = 'True')
    plt.plot(pred2, label = 'Predicted')
    plt.title('True data vs prediction')
    plt.legend()
    plt.show()
    
    corr = np.corrcoef(true2, pred2)
    sns.heatmap(corr, annot=True)
    plt.title('Correlation true data - prediction')
    plt.show()

    mae = metrics.mean_absolute_error(true2, pred2)
    mse = metrics.mean_squared_error(true2, pred2)
    rmse =  np.sqrt(mse)
    mape = np.mean(np.abs((true2 - pred2) / np.abs(true2)))
    acc = (1 - mape)
    
    print("\tr^2=%f"%metrics.r2_score(true2,pred2))
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*acc, 2))

    return mae, mse, rmse, mape, acc
# %%
def load_data(m_data):
    m_data=m_data[['Timestamp','Day_Ahead_price', 'DK_1_imports', 'SE_4_imports', 'DK_1_exports','SE_4_exports', 'Actual_Load','Solar_[MW]', 'ttf_price', 'coal_price', 'co2_price','DE_LU_AT_imports', 'DE_LU_AT_exports', 'Wind Total',"Hour","Date","Weekday","business"]]
    #m_data=m_data[['Timestamp','Day-ahead prices', 'Actual Load', 'Solar','Wind Total',"TTF","CO2","Hour","Day","business"]]
    return m_data
    
def create_time(data,days):
    sdate = data.min()
    edate = data.max() + pd.DateOffset(days)
    dt=pd.date_range(sdate,edate,freq='H')
    dl = pd.DataFrame(data=None)
    dl["Timestamp"]=dt
    return dl

def lagging(f1):
    cols=['DK_1_imports', 'SE_4_imports', 'DK_1_exports','SE_4_exports', 'Actual_Load','Solar_[MW]', 'DE_LU_AT_imports', 'DE_LU_AT_exports', 'Wind Total']
    for i in cols:
        f1[i+"_mean4"]=f1[i].shift(1, axis = 0).rolling(4,min_periods=1).mean().round(2)
        f1=f1.drop(i,axis=1)
    return f1


def split(datam):
    time = create_time(datam['Timestamp'],days=14)
    datam =time.merge(datam, how='left', on='Timestamp')
    datam["Weekday"]=datam["Timestamp"].dt.weekday
    datam["Hour"]=datam["Timestamp"].dt.hour
    datam=datam.drop_duplicates('Timestamp', keep='last')  
    
    dn = dict(tuple(datam.groupby('Weekday')))
    df0 = pd.DataFrame(data=None)
    for j in range(7):
        dn[j]=dict(tuple(dn[j].groupby('Hour')))
        for i in range(24):
            dn[j][i]=lagging(dn[j][i])
            if(i==0 and j==1):
                df0 =time.merge(dn[j][i], how='inner', on='Timestamp')
            else:
                df0=df0.append(dn[j][i], ignore_index=True)
    df0=df0.sort_values("Timestamp").reset_index(drop=True)
    cols=["co2_price","ttf_price","coal_price","Day_Ahead_price"]
    for i in cols:
        df0[i]=df0[i].fillna(method='pad')
    df0=df0.fillna(method='pad')
    out_data=df0
    return out_data
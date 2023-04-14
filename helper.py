import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression

import datetime as dt
import seaborn as sns

import xgboost as xg

def read_our_data(file_name):
    file_dir = './data/'+file_name
    df = pd.read_csv(file_dir)
    cols_check = ['Timestamp', 'Date']
    for col in cols_check:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

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

def build_baseline(x_train, y_train, x_test, y_test, timeframe):
    if timeframe == 'short_term':
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        model_ = lr
        ypred = model_.predict(x_test)
    return ypred, model_

def build_rft(x_train,y_train,x_test,y_test, random_search = False):
    # perform randomized search cross validation
    rf = RandomForestRegressor(random_state = 42, min_samples_split = 10)
    if random_search:
        rf_grid = {'max_depth': [None, 3, 5, 10],
                   'min_samples_split': np.arange(2, 20, 2),
                   'min_samples_leaf': np.arange(1, 20, 2)}
        rf = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 30, cv = 3, random_state=42, n_jobs = -1)
        rf.fit(x_train, y_train)
        # the optimal parameters
        rf.best_estimator_.get_params()
        # evaluate the optimal model
        model_ = rf.best_estimator_
    else:
        rf.fit(x_train, y_train)
        model_ = rf
        
    ypred = model_.predict(x_test)
    
    return ypred, model_

def build_xgb(x_train,y_train,x_test,y_test):
    xgb = xg.XGBRegressor(seed = 42)
    xgb.fit(x_train, y_train)
    
    model_= xgb
    ypred = model_.predict(x_test)

    return ypred, model_

def run_model(model_type, df, k_folds, split_method, train_start,
              features, target, cols_std, timeframe): # argument timeframe is necessary to choose baseline
    ypred = []
    models = []
    for k in range(k_folds):
        # split in train and test set
        train_set, test_set = split_timeseries(df, train_start, k, method = split_method)

        # get features and target
        X_train, y_train = get_feature_target(train_set, features, target)
        X_test, y_test = get_feature_target(test_set, features, target)

        # standardize
        X_train_std, X_test_std = standardize(X_train, X_test, cols_std)

        if model_type == 'baseline':
            yhat, model_ = build_baseline(X_train_std, y_train, X_test_std, y_test, timeframe = timeframe)
        elif model_type == 'rft':
            yhat, model_ = build_rft(X_train_std, y_train, X_test_std, y_test, random_search = True)
        elif model_type == 'xgb':
            yhat, model_ = build_xgb(X_train_std, y_train, X_test_std, y_test)
        elif model_type == 'nn':
            yhat, model_ = build_nn(X_train_std, y_train, X_test_std, y_test)

        print('Iteration ', k)
        model_evaluation(yhat, y_test)
        
        ypred.append(yhat)
        models.append(model_)
        
    return ypred, models

def model_evaluation(true, pred):

    plt.plot(true, label = 'True')
    plt.plot(pred, label = 'Predicted')
    plt.title('True data vs prediction')
    plt.legend()
    plt.show()
    
    corr = np.corrcoef(true, pred)
    sns.heatmap(corr, annot=True)
    plt.title('Correlation true data - prediction')
    plt.show()

    mae = metrics.mean_absolute_error(true, pred)
    mse = metrics.mean_squared_error(true, pred)
    rmse =  np.sqrt(mse)
    mape = np.mean(np.abs((true - pred) / np.abs(true)))
    acc = (1 - mape)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*acc, 2))

    return mae, mse, rmse, mape, acc
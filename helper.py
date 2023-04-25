import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import datetime as dt
import seaborn as sns

import xgboost as xg

#%%
def read_our_data(file_name):
    file_dir = './dataset_management/data/clean/'+file_name
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
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    model_ = lr
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

def build_rf(x_train, y_train, x_test, random_search = False):
    rf = RandomForestRegressor(random_state = 42, min_samples_split = 10)
    if random_search: # perform randomized search cross validation
        rf_grid = {'max_depth': [3, 7, 10],
                   'min_samples_split': np.arange(2, 30, 2),
                   'min_samples_leaf': np.arange(2, 15, 2)}
        rf = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 30, cv = 3, random_state = 42, n_jobs = -1)
        rf.fit(x_train, y_train)
        rf.best_estimator_.get_params() # optimal parameters
        model_ = rf.best_estimator_ # evaluate optimal model
    else:
        rf.fit(x_train, y_train)
        model_ = rf
        
    ypred = model_.predict(x_test)
    params = model_.get_params()
    
    return ypred, params

def build_xgb(x_train, y_train, x_test):
    xgb = xg.XGBRegressor(seed = 42)
    xgb.fit(x_train, y_train)
    
    model_= xgb
    ypred = model_.predict(x_test)
    params = model_.params()

    return ypred, params

def nn (x_train, y_train, x_test, y_test):

    model_ = nn
    ypred = model_.predict(x_test)
    return ypred, model_

def run_model(model_type, df, k_folds, split_method, train_start, features, target, cols_std, pred_window = []): # argument timeframe is necessary to choose baseline
    # model_type: lr (linear regression), ma (moving average), rf (random forest), xgb (XGBoost), nn (neural network)

    if model_type == 'ma':
        ypred = build_ma(df, target, pred_window)
        return ypred

    else:
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

            if model_type == 'lr':
                yhat, model_ = build_lr(X_train_std, y_train, X_test_std)
            elif model_type == 'rf':
                yhat, model_ = build_rf(X_train_std, y_train, X_test_std, random_search = True)
            elif model_type == 'xgb':
                yhat, model_ = build_xgb(X_train_std, y_train, X_test_std)
            elif model_type == 'nn':
                yhat, model_ = build_nn(X_train_std, y_train, X_test_std)

            print('Iteration ', k)
            model_evaluation(yhat, y_test)
            
            ypred.append(yhat)
            models.append(model_)
            
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
    
    corr = np.corrcoef(true2, pred)
    sns.heatmap(corr, annot=True)
    plt.title('Correlation true data - prediction')
    plt.show()

    mae = metrics.mean_absolute_error(true2, pred2)
    mse = metrics.mean_squared_error(true2, pred2)
    rmse =  np.sqrt(mse)
    mape = np.mean(np.abs((true2 - pred2) / np.abs(true2)))
    acc = (1 - mape)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    print('Accuracy:', round(100*acc, 2))

    return mae, mse, rmse, mape, acc
# %%



################### Features Importance

# From Fran
def imp_df(column_names, importances):
    """Function for creating a feature importance dataframe"""
    df = pd.DataFrame({'feature': column_names, 'feature_importance': importances})\
           .sort_values('feature_importance', ascending=False)\
           .reset_index(drop=True)
    return df

# From Fran
def var_imp_plot(imp_df, title):
    """Plotting a feature importance dataframe (horizontal barchart)"""
    sns.barplot(x='feature_importance', y='feature', data=imp_df, 
                orient='h', color='mediumpurple')\
       .set_title(title, fontsize = 20)

def lin_reg_feat_importance(x_train, y_train, x_test, y_test):
    
    # Scaling the data
    model_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_regression", LinearRegression())
    ])

    # R^2 score calculation
    model_lr.fit(x_train, y_train)
    model_lr.score(x_test, y_test)
    lr_imp = imp_df(x_train.columns, model_lr["linear_regression"].coef_)

    return lr_imp

def build_rf_2(x_train, y_train, x_test, random_search = False):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state = 42, min_samples_split = 10)
    if random_search: # perform randomized search cross validation
        rf_grid = {'max_depth': [3, 7, 10],
                   'min_samples_split': np.arange(2, 30, 2),
                   'min_samples_leaf': np.arange(2, 15, 2)}
        rf = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 30, cv = 3, random_state = 42, n_jobs = -1)
        rf.fit(x_train, y_train)
        rf.best_estimator_.get_params() # optimal parameters
        model_ = rf.best_estimator_ # evaluate optimal model
    else:
        rf.fit(x_train, y_train)
        model_ = rf
    
    feature_importances = model_.feature_importances_
    feature_names = list(x_train.columns)
    imp_df_ = imp_df(feature_names, feature_importances)
    var_imp_plot(imp_df_, "Feature importance for Random Forest")

    ypred = model_.predict(x_test)
    params = model_.get_params()
    
    return ypred, params


def gamw(x_train, y_train, random_search=False):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    rf = RandomForestRegressor(random_state=42, min_samples_split=10)
    if random_search:
        rf_grid = {
            'max_depth': [3, 7, 10],
            'min_samples_split': np.arange(2, 30, 2),
            'min_samples_leaf': np.arange(2, 15, 2)
        }
        rf = RandomizedSearchCV(
            estimator=rf, param_distributions=rf_grid, n_iter=30, cv=3, random_state=42, n_jobs=-1
        )
        rf.fit(x_train, y_train)
        model_ = rf.best_estimator_
    else:
        rf.fit(x_train, y_train)
        model_ = rf
        
    importances = model_.feature_importances_
    return importances



def xgb_feat_imp(x_train, y_train):
    
    import xgboost as xgb
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    xg_reg.fit(x_train,y_train)
    importances = xg_reg.feature_importances_
    return importances

# def lin_reg_feat_importance(x_train, y_train, x_test, y_test):
    
#     # Scaling the data
#     model_lr = Pipeline([
#         ("scaler", StandardScaler()),
#         ("linear_regression", LinearRegression())
#     ])

#     # R^2 score calculation
#     model_lr.fit(x_train, y_train)
#     model_lr.score(x_test, y_test)
#     lr_imp = imp_df(df.columns, model_lr["linear_regression"].coef_)

#     lr_imp

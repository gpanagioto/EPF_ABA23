import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# maybe not useful

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from operator import itemgetter


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

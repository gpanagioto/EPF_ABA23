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



def random_f_2(df, x_train, y_train, x_test, y_test):
    
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
        
    model_.fit(x_train, y_train)
    
    print("R^2 Training Score:", model_.score(x_train, y_train))
    print("R^2 OOB Score:", model_.oob_score_)
    print("R^2 Test Score:", model_.score(x_test, y_test))
    
    pred_diff = abs(y_train - model_.predict(x_train))
    min_error_ind = np.argmin(pred_diff)
    max_error_ind = np.argmax(pred_diff)
    print('Index with smallest error:', min_error_ind)
    print('Index with largest error:', max_error_ind)
    
    selected_rows = [min_error_ind, max_error_ind]
    selected_df = x_train.iloc[selected_rows, :].values
    prediction, bias, contributions = ti.predict(model_, selected_df)
    
    for i in range(len(selected_rows)):
        print("Row", selected_rows[i])
        print("Prediction:", prediction[i][0], 'Actual Value:', y_train[selected_rows[i]])
        print("Bias (trainset mean)", bias[i])
        print("Feature contributions:")
        for c, feature in sorted(zip(contributions[i], x_train.columns), 
                                key=lambda x: -abs(x[0])):
            print(feature, round(c, 2))
        print("-"*20) 
        
        
    prediction1, bias1, contributions1 = ti.predict(model_, 
                                                    np.array([selected_df[0]]), 
                                                    joint_contribution=True)
    prediction2, bias2, contributions2 = ti.predict(model_, 
                                                    np.array([selected_df[1]]), 
                                                    joint_contribution=True)
    
    aggregated_contributions1 = utils.aggregated_contribution(contributions1)
    aggregated_contributions2 = utils.aggregated_contribution(contributions2)
    
    res = []
    for k in set(aggregated_contributions1.keys()).union(
                set(aggregated_contributions2.keys())):
        res.append(([x_train.columns[index] for index in k], 
                aggregated_contributions1.get(k, 0) \
                    - aggregated_contributions2.get(k, 0)))   
            
    for lst, v in (sorted(res, key=lambda x:-abs(x[1])))[:10]:
        print (lst, v)
        
    return model_


def
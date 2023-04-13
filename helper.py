import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def split_timeseries(df, train_s, train_e, test_s, test_e): # format of dates has to be YYYY-mm-dd
    if (train_e <= train_s):
        print('Incoherent dates for the TRAIN set!')
    elif (test_e <= test_s):
        print('Incoherent dates for the TEST set!')
    else:
        X_train = df.loc[train_s:train_e]
        X_test = df.loc[test_s:test_e]
        
        return X_train, X_test
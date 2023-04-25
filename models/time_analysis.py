import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import pandas_profiling as pp
from scipy import stats 

import warnings
warnings.filterwarnings('ignore')


rcParams['figure.figsize'] = 10, 6

# function to visualize the price and the scatterplot of the prices 

def plot_n_scat (timeseries):
    plt.figure(figsize=(20,8))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Day-ahead prices')
    plt.plot(timeseries)
    plt.title('price actual')
    plt.show()
    
    plt.figure(figsize=(20, 8))
    df_price = timeseries
    df_price.plot(style='k.')
    plt.title('Scatter plot of price actual')
    plt.show()
    
    
def plot_hist(returns):
    plt.figure(figsize=(20,8))
    plt.axvline(returns.mean(),color="k",linestyle="dashed",linewidth=2)
    plt.axvline(returns.median(),color="r",linestyle="dashed",linewidth=2)
    plt.title("Histogram for prices")
    plt.xlabel("Time")
    plt.ylabel("Prices")
    plt.legend(["Mean","Median"])
    plt.figure(figsize=(20,10))
    sns.histplot(returns,kde=True, stat='density', label='Sample',bins=100)
    plt.show()
    
    
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='yellow',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)

# function of Seasonal decompose

def seasonal_decomp(df, ts_col):
    df_energy_new = df[ts_col].apply(lambda x: 0.01 if x < 0.01 else x)
    result = seasonal_decompose(df_energy_new, model='multiplicative', period=24)
    result.plot()
    plt.show()
    
def profiling(df,name:str):
    profile = pp.ProfileReport(df)
    profile.to_file(name)
    return

def split_data(df,show=False):
    train_data, test_data = df[3:int(len(df)*0.9)], df[int(len(df)*0.9):]
    if show==True:
        plt.figure(figsize=(20,8))
        plt.grid(True)
        plt.xlabel('time')
        plt.ylabel('value actual')
        plt.plot(train_data, 'green', label='Train data')
        plt.plot(test_data, 'blue', label='Test data')
        plt.legend()
    return train_data,test_data

def fit_moving_average(timeseries, lags):
    return list([np.average(timeseries.iloc[max(i-lags,0):i-1]) for i in range(1,len(timeseries)+1)])

def fit_arima_fun(timeseries,max_p,max_d,max_q,season,print_summary=False,exog=None):
    model_autoARIMA = auto_arima(timeseries, exog, start_p=0, start_q=0,
    test='adf',       # use adftest to find optimal 'd'
    max_p=max_p, max_q=max_q, # maximum p and q
    max_d=max_d,
    m=1,              # frequency of series
    d=None,           # let model determine 'd'
    seasonal=season,   # No Seasonality
    start_P=0, 
    D=0, 
    trace=True,
    error_action='ignore',  
    suppress_warnings=True, 
    stepwise=True)
    if print_summary==True:
        print(model_autoARIMA.summary())
    return model_autoARIMA 

def fit_arima_(df_energy1,column,max_p,max_d,max_q,hourly,season=False,exog=None,print_summary=True):
    order_p=[]
    order_q=[]
    predictions=[]
    if hourly==True:
        for i in range(24):
            cnt=i
            train_data, test_data=split_data(df_energy1[df_energy1.Hour==cnt][column])
            #acf_plot(df_energy_index['Day-ahead prices'],200)
            #slow train 
            print(i)
            best_mod = fit_arima_fun(train_data,max_p,max_d,max_q,seasonal=season,exog=exog,print_summary=True)
            p,d,q=best_mod.order
            order_p.append(p)
            order_q.append(q)
            predicted_mu = best_mod.predict(n_periods=7)
            #plotting(test_data[:7].reset_index(), predicted_mu, 0, train_data)
            predictions.append(predicted_mu)
        
    if hourly== False:
        train_data, test_data=split_data(df_energy1[column])
        #acf_plot(df_energy_index['Day-ahead prices'],200)
        #slow train 
        best_mod = fit_arima_fun(train_data,max_p,max_d,max_q,seasonal=season,exog=exog,print_summary=True)
        p,d,q=best_mod.order
        order_p.append(p)
        order_q.append(q)
        predicted_mu = best_mod.predict(n_periods=24)
        #plotting(test_data[:24], predicted_mu, 0, train_data)
        predictions.append(predicted_mu)
    return order_p,order_q,predictions,test_data


def test_arima_(df_energy1,xtest,column,steps,hourly,season=False,exog=None):
    order_p=[]
    order_q=[]
    predictions=[]
    if hourly==True:
        for i in range(24):
            cnt=i
            train_data = (df_energy1[df_energy1.Hour==cnt][column])
            #acf_plot(df_energy_index['Day-ahead prices'],200)
            #slow train 
            best_mod = fit_arima_fun(train_data,max_p,max_d,max_q,seasonal=season,exog=exog,print_summary=True)
            p,d,q=best_mod.order
            order_p.append(p)
            order_q.append(q)
            predicted_mu = best_mod.predict(n_periods=steps)
            #plotting(test_data[:7].reset_index(), predicted_mu, 0, train_data)
            predictions.append(predicted_mu)
        
    if hourly== False:
        for l in xtest[column].index:
            data=df_energy1[column][df_energy1.index<l]
            if exog!=None:
                exog=df_energy1[['DK_1_imports', 'SE_4_imports', 'DK_1_exports','SE_4_exports','Forecasted_Load', 'Actual_Load','Solar_[MW]', 'ttf_price', 'coal_price', 'co2_price','Biomass_Actual_Aggregated_[MW]', 'Waste_Actual_Aggregated_[MW]','DE_LU_AT_imports', 'DE_LU_AT_exports', 'Wind Total']][df_energy1.index<l]
            model=ARIMA(data.values,exog, order=(8, 0, 1)) #Use precalculated orders
            arima=model.fit()
            pred=arima.predict(n_periods=steps )
            print(arima.summary())
            break

    return pred[:steps]



def acf_plot(timeseries,lag):
    plt.rc("figure", figsize=(20, 8))
    plot_acf(timeseries, lags = lag)
    plt.title("autocorrelation")
    plot_pacf(timeseries, lags = lag)
    plt.title("partial autocorrelation")
    plt.show()
    
    
def plotting(test, predicted, flag, train=None):
    if flag==1:
        plt.plot(train, 'green', label = 'training prices')
    plt.plot(test, 'orange', label = 'test prices')
    plt.plot(predicted, 'blue', label = 'predicted prices')
    plt.legend(loc = 'upper left', fontsize = 8)
    plt.show()

def box(df,column):
    #Creation of Box Plot
    df[column].to_frame().boxplot(figsize=(15,8))
    plt.title(column)
    plt.ylabel('Values')
    plt.show()
    
def qqplot(y):
    theoretical_qt = np.random.normal(0,1, len(y))
    # Specify axes
    x = sorted(y)
    y = sorted(theoretical_qt)
    # Plotting
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    plt.scatter(x, y, s=15)
    ax.set_xlabel('OC Quantiles')
    ax.set_ylabel('Normal Quantiles')
    sm.qqline(ax, "r", x, y) # Adding 45Deg line
    ax.grid()
    plt.show()
        
# this function provides the Kurtosis, Skewness and Jarque Bera test for test of normality.
#Kurtosis and Skewness of normal distribution is 3 and 0 respectively.
def testnormal(data):
    print("Kurtosis of",str(data.name)+": " ,stats.kurtosis(data))
    print("Skewness of",str(data.name)+": " ,stats.skew(data))
    print("Jarque-Bera Result: " + str(stats.jarque_bera(data))) # The Jarque–Bera test of normality compares the sample skewness and kurtosis to 0 and 3, their values under normality.


def kpss_test(y):
    kpss_test = kpss(y, regression='ct')
    print('KPSS Statistic: {:.6f}\np-value: {:.6f}\n#Lags used: {}'
        .format(kpss_test[0], kpss_test[1], kpss_test[2]))
    for key, value in kpss_test[3].items():
        print('Critical Value ({}): {:.6f}'.format(key, value))
    


def req():
    from entsoe import EntsoePandasClient
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    import datetime as dt
    f=pd.DataFrame(data=None)
    client = EntsoePandasClient(api_key="5904b3c4-7835-4b64-a741-480681944340")
            #price
    START_DATE = '20230301'
    END_DATE = dt.datetime.now().date().strftime("%Y%m%d")
    start = pd.Timestamp(START_DATE, tz = 'Europe/Copenhagen')
    end = pd.Timestamp(END_DATE, tz = 'Europe/Copenhagen')
    # countries
    COUNTRY_CODE = 'DK_2' # Denmark (Copenhagen)
    #query_installed_generation_capacity
    f=client.query_generation_per_plant(COUNTRY_CODE, start=start,end=end)



def create_quarterly_data(df):
    r=df.Day_Ahead_price.copy()
    r.index=df.Timestamp
    r=r[~r.index.duplicated(keep='first')]
    r=r.resample('15Min')
    r=r.interpolate(method='pad').astype('int')
    r=r[-24*4:]
    r.to_csv('dataset_management\data\clean\clean_quarterly.csv',index=False)
    return r
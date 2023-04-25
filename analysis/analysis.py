import numpy as np
import pandas as pd


# This function creates new dataframes, grouped by hour and the mean of the attribute. 
# Passing in the dataframe, the year and the desired attribute, will return the necessary dataframe

def split_dataset(df, year_s, attribute):
    df_year_s = df[df['Year'] == year_s]
    df_year_s_gr = df_year_s.groupby('Hour')[attribute].mean().reset_index().set_index('Hour')
    return df_year_s_gr



# This function takes as an input a dataframe, a year and a feature. Combined with the function 'split_dataset', it creates 
# a plot of the desired feature, throughout the 24-hour frame

def plot_hourly_data(df, year, attribute):
    import matplotlib.pyplot as plt
    
    df_year_attribute = split_dataset(df, year, attribute)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(16,6))
    
    plt.plot(df_year_attribute.index, df_year_attribute[attribute], color='blue')
    
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel(attribute, fontsize=14)
    plt.title(f'Hourly {attribute} for year {year}', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.legend()
    
    plt.show()



# A function that receives as input a a dataframe, a year and a feature. Combined with the function 'split_dataset' and a list of years, 
# it will create a single plot with all the lines of all the desired years.

def plot_hourly_data_all_years(df, years, attribute):
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16,6))
    for year in years:
        df_year_attribute = split_dataset(df, year, attribute)
        plt.plot(df_year_attribute, label=year, linewidth=1)
    plt.xlabel('Hour of the day', fontsize=14)
    plt.ylabel(attribute, fontsize=14)
    plt.title(f'Hourly {attribute} for all years', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()
    
    return





def calculate_load_difference(df, year):
    df_year = split_dataset(df, year, 'Actual_Load')
    df_year['Forecasted_Load'] = split_dataset(df, year, 'Forecasted_Load')
    df_year['Load_Difference'] = df_year['Actual_Load'] - df_year['Forecasted_Load']
    return df_year


def plot_load_difference(df, year):
    
    import matplotlib.pyplot as plt
    
    df_year = calculate_load_difference(df, year)
    
    fig, ax = plt.subplots(figsize=(22, 10))
    ax.stackplot(df_year.index, [df_year['Actual_Load'], df_year['Forecasted_Load'], df_year['Load_Difference']],
                 labels=['Actual Load', 'Forecasted Load', 'Load Difference'],
                 colors=['#455d7a', '#f95959', '#42b883'])
    
    ax.set_xlabel('Hour of the day')
    ax.set_ylabel('Load [MW]')
    ax.set_title(f'Hourly Load for year {year}')
    ax.legend(loc='upper left')
    plt.show()
    
    return



# This function gets as input parameters a dataframe, two different years and an attribute of the dataframe. It then creates a dataframe,
# that its features are the values of the attribute for these two years, as well as their difference.

def compare_years(df, year_1, year_2, attribute):
    df_year_1 = split_dataset(df, year_1, attribute)
    df_year_2 = split_dataset(df, year_2, attribute)
    df_diff = pd.concat([df_year_1[attribute], df_year_2[attribute]], axis=1)
    df_diff.columns = [f"{year_1}_{attribute}", f"{year_2}_{attribute}"]
    df_diff[f"{year_1}_{year_2}_{attribute}_diff"] = df_year_2[attribute].reset_index(drop=True) - df_year_1[attribute].reset_index(drop=True)
    return df_diff



import pandas as pd
import datetime as dt
import numpy as np
import os
import re
import argparse
from entsoe import EntsoePandasClient
import configparser
import time
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

api_key = config["api_key"]
#print(api_key)

client = EntsoePandasClient(api_key = api_key)

#dates
START_DATE = '20180101'
END_DATE = '20221231'

# countries
COUNTRY_CODE = 'DK_2' # Denmark (Copenhagen)


# the path where the data will be saved
SAVE_PATH = "data/"

def get_arguments():
    parser = argparse.ArgumentParser(description='retrieving data from Entsoe \
                                     for the EPF Project')
    parser.add_argument('--country_code', type=str, default=COUNTRY_CODE,
                          help='')
    parser.add_argument('--start_date',type=str, default=START_DATE,
                        help='Data period starts this date')
    parser.add_argument('--end_date',type=str, default=END_DATE,
                        help='Data ends period ends this date')    
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the data')
    return parser.parse_args()

def main():
    
    start_time = time.time() 

    args = get_arguments()

    # period
    start = pd.Timestamp(args.start_date, tz = 'Europe/Copenhagen')
    end = pd.Timestamp(args.end_date, tz = 'Europe/Copenhagen')
    country = args.country_code
    save_path = args.save_path

    if not os.path.exists('data/' + country):
        os.makedirs('data/' + country)
    save_path = 'data/' + country + '/'
    print(save_path)


    print(f"Data Period: {start} - {end}.")
    print(f"Country: {country}.")

    print("\n --------- Start Scraping --------- \n")

    # 
    day_ahead_prices = client.query_day_ahead_prices(args.country_code, start = start, end = end)
    day_ahead_prices.to_csv(save_path + 'day_ahead_prices.csv')
    print(f"Size Day Ahead Prices is {day_ahead_prices.shape[0]} rows.\n")

    #
    load_and_forecast = client.query_load_and_forecast(args.country_code, start = start, end = end)
    load_and_forecast.to_csv(save_path + 'load_and_forecast.csv')
    print(f"Size Load And Forecast is {load_and_forecast.shape[0]} rows.\n")

    #
    wind_solar_forecast = client.query_wind_and_solar_forecast(args.country_code, start = start, end = end)
    wind_solar_forecast.to_csv(save_path + 'wind_solar_forecast.csv')
    print(f"Size Wind Solar Forecast is {wind_solar_forecast.shape[0]} rows.\n")


    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()
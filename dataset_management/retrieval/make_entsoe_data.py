import pandas as pd
import datetime as dt
import numpy as np
from data_retrieving import EntsoeDataRetrieval, Directory
import argparse
from entsoe import EntsoePandasClient
import time
import yaml
import os

# Get the absolute path to the project directory of the script
project_dir = os.path.abspath(os.path.join(os.path.dirname('EPF_ABA23'), '..'))

with open(project_dir+"/dataset_management/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

api_key = config["api_key"]

client = EntsoePandasClient(api_key = api_key)

#dates
START_DATE = '20171224'
END_DATE = dt.datetime.now().date().strftime("%Y%m%d")

# countries
COUNTRY_CODE = 'DK_2' # Denmark (Copenhagen)

# the path where the data will be saved
SAVE_PATH = project_dir+"/dataset_management/data/raw/entsoe"
print(SAVE_PATH)

def get_arguments():
    parser = argparse.ArgumentParser(description='retrieving data from Entsoe \
                                     for the EPF Project')
    parser.add_argument('--data_type', type=str, default=None,
                          help='The type of data we want ["main", "imports", "exports"]')
    parser.add_argument('--query', type=str, default=None,
                          help='The data query in case of main data choices=["DayAheadPrices", "LoadAndForecast", "WindSolarForecast"]')    
    parser.add_argument('--country_code', type=str, default=COUNTRY_CODE,
                          help='The country we want the data for')
    parser.add_argument('--country_to', type=str, default=None,
                          help='The country the exports go to choices "DE_LU"/"DK_1"/"SE_4"/"DE_LU_AT"')
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
    data_type = args.data_type
    query = args.query

    data_retrieving = EntsoeDataRetrieval(client=client, start_date=start, end_date=end, country_code=country)
    
    print(f"Data Period: {start} - {end}.")
    print("\n --------- Start Scraping --------- \n")

    if data_type == 'imports':
        data_save_path = Directory(save_path , data_type, country)
        data_retrieving.Imports(data_save_path)

    elif data_type == 'exports':
        country_to = args.country_to
        data_save_path = Directory(save_path, data_type, country)
        print(f"Exports data to {country_to}")
        data_retrieving.Exports(data_save_path, country_to=country_to)

    else:
        data_save_path = Directory(save_path, data_type, country)
        print(f"{query} data for {country}")

        if query == "DayAheadPrices":
           data_retrieving.DayAheadPrices(data_save_path)
        elif query == "LoadAndForecast":
            data_retrieving.LoadAndForecast(data_save_path)
        else:
            data_retrieving.WindSolarForecast(data_save_path)
          
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))
    
if __name__ == '__main__':
    main()
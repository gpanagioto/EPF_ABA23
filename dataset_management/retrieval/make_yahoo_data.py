import yfinance as yf
import time
import argparse
import datetime as dt
from data_retrieving import YahooDataRetrieval, Directory
import pandas as pd
import os

# Get the absolute path to the project directory of the script
project_dir = os.path.abspath(os.path.join(os.path.dirname('EPF_ABA23'), '..'))

#dates
START_DATE = "2018-01-01"
END_DATE = dt.datetime.now().date().strftime("%Y-%m-%d")

# the path where the data will be saved
SAVE_PATH = project_dir+"/dataset_management/data/raw/"

def get_arguments():
    parser = argparse.ArgumentParser(description='retrieving data from Yahoo API \
                                     for the EPF Project')
    parser.add_argument('--ticker', type=str, default=None,
                          help='The ticker we want the data for \
                                [ \
                                 "TTF=F : Dutch TTF Natural Gas Calendar", \
                                 "" :, \
                                 "" : \
                                ]')
    parser.add_argument('--start_date',type=str, default=START_DATE,
                        help='Data period starts this date')
    parser.add_argument('--end_date',type=str, default=END_DATE,
                        help='Data ends period ends this date')    
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the data')
    return parser.parse_args()
#
 
def main():

    start_time = time.time()
    # using sleep() to hault the code execution
    time.sleep(6)
 
    args = get_arguments()

    start = pd.Timestamp(args.start_date, tz = 'Europe/Copenhagen')
    end = pd.Timestamp(args.end_date, tz = 'Europe/Copenhagen')
    ticker = args.ticker
    save_path = args.save_path

    data_save_path = Directory(save_path, 'yahoo', 'TTF')
    data_retrieving = YahooDataRetrieval(data_save_path, ticker, start_date=start, end_date=end)
    
    print(f"Data Period: {start} - {end}.")
    print("\n --------- Start Scraping --------- \n")

    print("\nThe time of code execution end is : ", time.ctime())

if __name__ == '__main__':
    main()
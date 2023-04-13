import pandas as pd
import datetime as dt
import numpy as np
import os
import re
import argparse
from entsoe import EntsoePandasClient
import configparser

client = EntsoePandasClient(api_key = '5904b3c4-7835-4b64-a741-480681944340')

start = pd.Timestamp('20180101', tz = 'Europe/Copenhagen')
end = pd.Timestamp('20221231', tz = 'Europe/Copenhagen')

# countries
country_code = 'DK_2' # Denmark (Copenhagen)
country_code_from = 'DE_LU' # Germany - Luxemburg
country_code_to = 'DK_2' # East Denmark

def get_arguments():
    parser = argparse.ArgumentParser(description='retrieving data from Entsoe \
                                     for the EPF Project')
    parser.add_argument('--country_code', type=str, default=country_code,
                          help='')
    parser.add_argument('--country_code_from', type=str, default=country_code_from,
                          help='')
    parser.add_argument('--country_code_to',type=str,default=country_code_to,
                        help='')
    parser.add_argument('--start_date',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()

def Retrieval(start, end, country_code, country_code_from, country_code_to):
    # period
    start = pd.Timestamp('20180101', tz = 'Europe/Copenhagen')
    end = pd.Timestamp('20221231', tz = 'Europe/Copenhagen')

    # countries
    country_code = 'DK_2' # Denmark (Copenhagen)
    country_code_from = 'DE_LU' # Germany - Luxemburg
    country_code_to = 'DK_2' # East Denmark
import pandas as pd
from merging import DataMerging
import yaml
import os
'''

def remove_utc(col_name, # str: name of the column that contains the object to convert to timestamp
               tz_offset, # int: timezone offset. E.g., CET = +1
               df # dataframe: contains all the info
              ):
    df['Timestamp'] = pd.to_datetime(df[col_name],utc = True)#, format = '%Y-%m-%d %H:%M:%S')
    df['Timestamp'] = (df['Timestamp'] + dt.timedelta(hours = tz_offset)).dt.tz_localize(None)
    df.drop([col_name], axis = 1, inplace = True) # drop the column
    df.set_index('Timestamp', inplace = True) # set column 'Timestamp' as index
    return df

'''

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

files = config["files_names"]

search_path = 'data/'
save_path = 'data/merge'

def main(search_path: str, save_path: str, files: dict):

    merging = DataMerging(search_path)

    final = pd.DataFrame()

    for file, data in files.items():

        if file == 'entsoe':
            print('-----Entsoe------')

            for key, file_name in data.items():
                print(key, ' : ' ,file_name)
                                
                df1 = merging.entsoe_timestamp(file_name) 

                final = pd.merge(final, df1, left_index=True, right_index=True ,how='outer')
 
        else:

            print('------- Yahoo -------')

            for key, file_name in data.items():
                print(key, ' : ' ,file_name)
                
                df2 = merging.yahoo_timestamp(file_name)

                final = pd.merge(final, df2, left_index=True, right_index=True ,how='outer')

        if not os.path.exists(save_path + '/'  ):
            os.makedirs(save_path + '/' )
    
        data_save_path = save_path + '/' 
    final.to_csv(data_save_path + 'merged.csv', index=True) 

if  __name__ == '__main__':
    main(search_path,save_path,files)



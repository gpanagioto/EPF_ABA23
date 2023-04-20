import pandas as pd
from merging_code import DataMerging, find_folder
import yaml
import os

# Get the absolute path to the project directory of the script
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

with open(project_dir+"/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

files = config["files_names"]

search_path = find_folder('raw', project_dir)
save_path = find_folder('data', project_dir)

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

        if not os.path.exists(save_path + '/merged/'  ):
            os.makedirs(save_path + '/merged/' )
    
        data_save_path = save_path + '/merged/'
        print(data_save_path)
    final.to_csv(data_save_path + 'merged.csv', index=True) 

if  __name__ == '__main__':
    main(search_path,save_path,files)



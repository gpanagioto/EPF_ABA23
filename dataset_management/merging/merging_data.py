import pandas as pd
from merging_code import DataMerging, find_folder
import yaml
import os
import datetime as datetime

# Get the absolute path to the project directory of the script
project_dir = os.path.abspath(os.path.dirname('EPF_ABA23'))
#print(project_dir)

config_file = os.path.join(project_dir,'dataset_management','config.yaml')
#print(config_file)

with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

files = config["files_names"]

search_path = find_folder('raw', project_dir)
save_path = find_folder('data', project_dir)
print(save_path)
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
 
        elif file == 'yahoo':
            print('-----Yahoo------')

            for key, file_name in data.items():
                print(key, ' : ' ,file_name)
                
                df2 = merging.yahoo_timestamp(file_name)

                final = pd.merge(final, df2, left_index=True, right_index=True ,how='outer')

        else:
            print('-----Energy Data------')
            
            df_conc = pd.DataFrame()

            for file_name in os.listdir(os.path.join(search_path,file)):
            
                print(file_name)

                df3 = merging.energy_data_timestamp(file_name)

                df_conc = pd.concat([df_conc, df3], ignore_index=False)    

            final = pd.merge(final, df_conc, left_index=True, right_index=True, how='outer')
                                      

        if not os.path.exists(save_path + '/merged/'  ):
            os.makedirs(save_path + '/merged/' )
    
        data_save_path = save_path + '/merged/'
        print(f'The save path is {data_save_path}.')

    final[final.index <= datetime.datetime.now()].to_csv(data_save_path + 'merged.csv', index=True) 

if  __name__ == '__main__':
    main(search_path,save_path,files)



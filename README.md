Electricity Price Forecasting

This repository contains the final project for the Advanced Business Analytics course at DTU, focused on electricity price forecasting. In this project, we aim to develop a predictive model to forecast electricity prices, leveraging historical data and machine learning techniques.

Project Objective

The main objective of this project is to create an accurate and robust forecasting model that can predict electricity prices with a high level of accuracy. The project will involve the following steps:

1.Data Collection: Gathering historical electricity price data from reliable sources.
2.Data Preprocessing: Cleaning, transforming, and preparing the data for analysis.
3.Exploratory Data Analysis (EDA): Conducting a thorough analysis of the data to gain insights and identify patterns.
4.Feature Engineering: Selecting relevant features or creating new features to improve the predictive power of the model.
5.Model Selection: Evaluating and selecting the most appropriate machine learning algorithms for the task.
6.Model Training: Training the selected models using historical data and optimizing hyperparameters.
7.Model Evaluation: Assessing the performance of the trained models using appropriate evaluation metrics.
8.Model Interpretation: Interpreting the results of the models and understanding their predictive power.
9.Model Deployment: Deploying the final forecasting model for real-time predictions.

Repository Structure

The repository is structured as follows:

* dataset_management: This directory contains the historical electricity price data and the code for both their retrieval and merge. 
It is also structured in the following way
    * Retrieval: The retrieval folder contains the code for retrieving data from NTSOE and Yahoo finance.
        * Entsoe
        
            --data_type DATA_TYPE           The type of data we want ["main", "imports", "exports"]
            
            --query QUERY                   The data query in case of main data: choices=["DayAheadPrices", "LoadAndForecast", "WindSolarForecast"]
            
            --country_code COUNTRY_CODE     The country we want the data for. Default: 'DK_2'
            
            --country_to COUNTRY_TO         The country the exports go to and applies only to 'exports': choices "DE_LU"/"DK_1"/"SE_4"/"DE_LU_AT"
            
            --start_date START_DATE         Data date from. Default: '20171224'
            
            --end_date END_DATE             Data date to. Default: Today
            
            --save_path SAVE_PATH           The directory to store the data

        Example: Create the data Day Ahead Prices:

        python make_entsoe_data.py --data_type 'main' --query 'DayAheadPrices'

        * Yahoo
    
            --ticker TICKER             The ticker we want the data for [ "TTF=F : Dutch TTF Natural Gas Calendar", "" :, "" : ]
            
            --start_date START_DATE     Data date from. Default: '20171224'
            
            --end_date END_DATE         Data date to. Default: Today
            
            --save_path SAVE_PATH       The directory to store the data

        Example: Create the data for Dutch TTF Natural Gas Calendar:

        python make_yahoo_data.py --ticker 'TTF=F'

        For CO2 prices and Coal prices it doesn't work as the yahoo finance doesn't have data.

    * Merging:  The merging folder contains the code for merging the retrieved data.

        python merging_data.py   

    * Cleaning: The cleaning folder the code for cleaning the merged data. 
    * Data: The data folder contains the raw data produced by the code in the retrieval folder, the merged data from merging and also the clean data.

 

* notebooks/: This directory contains Jupyter notebooks that document the various stages of the project, including data preprocessing, EDA, model training, evaluation, and interpretation.

* models/: This directory contains the trained machine learning models in serialized format, ready for deployment.
* reports/: This directory contains any reports, presentations, or visualizations generated as part of the project.
* src/: This directory contains any source code, scripts, or utilities developed for the project, such as data preprocessing scripts, model training scripts, or deployment scripts.
* README.md: This file, providing documentation about the project, including its objective, repository structure, and usage instructions.

Dependencies

The following dependencies are required to run the code in this repository:

* Python 3.x
* Jupyter Notebook or Jupyter Lab
* Data manipulation and visualization libraries such as NumPy, Pandas, and Matplotlib.
* Machine learning libraries such as Scikit-Learn, XGBoost, or TensorFlow, depending on the chosen model.
* Any additional dependencies specific to the project, such as time series libraries or custom packages.

Getting Started

To get started with this project, follow these steps:

1.Clone the repository to your local machine using git clone or by downloading the ZIP file.
2.Install the necessary dependencies as mentioned in the "Dependencies" section.
3.Explore the Jupyter notebooks in the notebooks/ directory, which document the different stages of the project.
4.Follow the instructions in the notebooks to preprocess the data, conduct EDA, train and evaluate machine learning models, and interpret the results.
5.Optionally, deploy the trained model using the code and scripts provided in the src/ directory.
6.Generate any reports, presentations, or visualizations in the reports/ directory to summarize the project findings.
7.Modify the README.md file to update the documentation with any changes or additional instructions.

Contributors

This project was developed by the following contributors:

Your Name: Role/Contribution
Feel free to contact the contributors or the course instructor for any questions, feedback, or suggestions regarding this project.

License

This project is released under the MIT License, which is an

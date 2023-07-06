import zipfile

import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as tsa
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import zipfile
##############################################################################################

## intitialize the Data Ingestion configuration

@dataclass
class DataIngestionconfig:
    #train_data_path = os.path.join('artifacts', 'train.csv')
    #test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw_data.zip')

##Create a data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df=pd.read_csv(os.path.join('notebooks/data','title_transactions-06-29-2021.csv.gz'))
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            csv_data = df.to_csv(index=False)
            #df.to_csv(self.ingestion_config.raw_data_path,index=False)

            with zipfile.ZipFile(self.ingestion_config.raw_data_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr('raw_data.csv', csv_data)

            logging.info('Raw data is created')

            logging.info('Data Transformation Started')
            df_cumsum_1, top_ten_counties = self.initiate_data_transformation(df=df)

            logging.info('Data Transformation finished')

                                    
            try:
                county_information = {}
                for county in top_ten_counties:
                    county_information[county] = {}
                    county_information[county]['df']=self.separate_data_by_county(county, df_cumsum_1)

            except Exception as e:
                logging.info('Exception occured at county_information_dict Stage')
                raise CustomException(e,sys)

            #splitting dataset into train and test sets for validation
            #logging.info('splitting dataset into train and test sets for validation')
            #print('splitting dataset into train and test sets for validation \n')

            #train_var = 'train_' + county
            #test_var = 'test_' + county

            #locals()[train_var], locals()[test_var] = self.train_test_split_ts(county_information[county]['df'], 0.80, 0.20)


            #train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.raw_data_path,
                county_information
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion Stage')
            raise CustomException(e,sys)
        

    def separate_data_by_county(self,county, df_cumsum_1):
        
        logging.info('separate_data_by_county starts')

        try:
            df_county = pd.DataFrame(df_cumsum_1.loc[:, county])
            df_county.columns=["EV's on the Road"]
            df_county.head()
            return df_county
        
        except Exception as e:
            logging.info('Exception occured at separate_data_by_county Stage')
            raise CustomException(e,sys)

 

   
            
################################################################################################
# 
#    
    def initiate_data_transformation(self, df):
        try:
            logging.info('Data Transformation initiated')

            #importing the data back into the notebook
            #df=pd.read_csv(r'C:\Users\pmoff\OneDrive\Desktop\PWSkills\MLProjects\EV_E_Mobility\.2_Electric_Vehicles_and_Chargers_in_Washington_State\Input_files\title_transactions-06-29-2021.csv.gz', compression='gzip', index_col=0)

            # Feature Engineering
            # Changing data type of 'transaction_date' to datetime
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])

            # Feature Engineering "m/y" column
            df['m/y'] = df['transaction_date'].dt.strftime("%m-%Y")

            # Set transaction_date as the index of df
            df.set_index('transaction_date', inplace=True)

            # Dropping columns that are for Governmental Use
            drop_cols = ['electric_vehicle_fee_paid',
                        'hb_2042_clean_alternative_fuel_vehicle_cafv_eligibility',
                        'meets_2019_hb_2042_electric_range_requirement', 
                        'meets_2019_hb_2042_sale_date_requirement', 
                        'meets_2019_hb_2042_sale_price_value_requirement', 
                        'transportation_electrification_fee_paid', 
                        'hybrid_vehicle_electrification_fee_paid', 'legislative_district', 
                        'non_clean_alternative_fuel']
            df.drop(drop_cols, axis=1, inplace=True)

            # Remove columns that contain repetitive or duplicate information 
            drop_cols = ['transaction_year', 'base_msrp', 'date_of_vehicle_sale']
            df.drop(drop_cols, axis=1, inplace=True)

            #filtering out registration information to keep vehicle purchase information
            df=df[(df['transaction_type']=='Original Title') | 
                (df['transaction_type']=='Transfer Title')]

            #Handling duplicates
            df.drop_duplicates(inplace=True)

            # Duplicates by Date/ID/County
            #df[df.duplicated(subset=['m/y', 'dol_vehicle_id', 'county'], keep=False)].sort_values('dol_vehicle_id')
            df.drop_duplicates(subset=['m/y', 'dol_vehicle_id', 'county'], keep='last', inplace=True)

            # Duplicates by Date/ID
            #df[df.duplicated(subset=['m/y', 'dol_vehicle_id'], keep=False)].sort_values('dol_vehicle_id')
            df.drop_duplicates(subset=['m/y', 'dol_vehicle_id'], keep='last', inplace=True)

            # Checking for and Addressing Null Values
            #dropping null values
            df.dropna(subset=['county'], inplace=True)

            #Vehicles owned by non-Washington residents will be excluded from the data analysis
            df=df[df['state_of_residence']=='WA']

            #Try to fill NA values
            #filling null values
            df['city'].fillna('Unknown', inplace=True)
            df['zip'].fillna('Unknown', inplace=True)

            # Data Standardization
            #Format the model names we can use map() method. ('Panamera' and 'PANAMERA')
            df['model']=df['model'].map(lambda x: x.title())
            df[df['model']=='Sonata Plug-In Hybrid']['electric_vehicle_type'].unique()
            df[df['model']=='Prius Plug-In']['electric_vehicle_type'].unique()

            df['model'].replace(['Niro Electric', 'Niro Plug-In Hybrid'], 'Niro', 
                                inplace=True)
            df['model'].replace(['Prius Plug-In', 'Prius Plug-In Hybrid'], 'Prius', 
                                inplace=True)
            df['model'].replace('Kona Electric', 'Kona', inplace=True)
            df['model'].replace('Optima Plug-In Hybrid', 'Optima', inplace=True)
            df['model'].replace('Sonata Plug-In Hybrid', 'Sonata', inplace=True)
            df['model'].replace('Xc60 Awd Phev', 'Xc60 Awd', inplace=True)
            df['model'].replace('Xc90 Awd Phev', 'Xc90 Awd', inplace=True)

            #Strore df in another variable
            data = df

            #We can slice out the new car purchases for each county and parse the information into a dictionary.
            county_dict = {}
            for county in list(data['county'].unique()):
                county_dict[county] = data[
                    (data['county']==county)&
                    (data['transaction_type']=='Original Title')
                ].resample('M').size().cumsum()

            #filling null values created by resampling with 0's.
            df_cumsum = pd.DataFrame(county_dict)
            df_cumsum.fillna(0, inplace=True)

            # Electric Vehicles on the Road in Washington State Over Time
            #summing each row to get monthly statewide total 
            df_cumsum['State Total'] = df_cumsum.sum(axis=1)
            df_cumsum = df_cumsum.reset_index()

            # Electric Vehicles on the Road Over Time by County
            top_ten_counties = data['county'].value_counts().head(10).index.tolist()

            #filtering out counties to keep top 10
            df_cumsum = df_cumsum.loc[:, ['transaction_date', *top_ten_counties]]

            #Strore df in another variable
            df_cumsum_1 = df_cumsum

            df_cumsum_1 = df_cumsum_1.set_index('transaction_date').resample('M').asfreq()
            #resample('M').asfreq() is used to resample the DataFrame df_cumsum at a monthly frequency.

            return (
                df_cumsum_1, top_ten_counties
            )
        
        except Exception as e:
            logging.info('Exception occured at Train test split Stage')
            raise CustomException(e,sys)

    


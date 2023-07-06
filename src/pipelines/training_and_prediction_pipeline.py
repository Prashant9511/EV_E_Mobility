

import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion_transformation import DataIngestion


if __name__=='__main__':
    obj=DataIngestion()
    raw_data_path,county_information=obj.initiate_data_ingestion()
    print(raw_data_path, county_information)
    
    model_trainer=ModelTrainer()
    model_trainer.model_train_and_predict(county_information)
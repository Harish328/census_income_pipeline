import os
import sys
from scr.exception import CustomException
from scr.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from scr.components.data_transformation import DataTransformation

## Initialize the Data Ingestion Configuration

@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def intitiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')

        try:
            df=pd.read_csv('notebook/data/gemstone.csv')
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test split')
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=35)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header = True)

            logging.info('Ingestion of data is completed')


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info('Exception occured in data ingestion step')
            raise CustomException(e,sys)



## run Data Ingetion

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path , test_data_path = obj.intitiate_data_ingestion()
    data_transformation =DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
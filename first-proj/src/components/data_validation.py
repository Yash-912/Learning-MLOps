import json 
import sys
import os 

import pandas as pd 
from pandas import DataFrame
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise MyException(e.sys)
        
    def validate_number_of_columns(self, dataframe : DataFrame) -> bool:
        try:
            status=len(dataframe.columns)==len(self._schema_config["columns"])
            logging.info(f"Are required columns present {status}")
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    def is_column_exist(self, df:DataFrame) -> bool:
        try:
            dataframe_cols=df.columns
            missing_num_cols=[]
            missing_cat_cols=[]
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_cols:
                    missing_num_cols.append(column)
                
            if len(missing_num_cols)>0:
                logging.info(f"Missing Numerical columns: {missing_num_cols}")
            
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_cols:
                    missing_cat_cols.append(column)
                
            if len(missing_cat_cols)>0:
                logging.info(f"Missing categorical columns: {missing_cat_cols}")
                
            return False if len(missing_num_cols)>0 or len(missing_cat_cols)>0 else True
        
        except Exception as e:
            raise MyException(e.sys)
    
    @staticmethod
    def read_file(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_err_msg=""
            logging.info("Starting Data validation")
            train_df, test_df=(DataValidation.read_file(file_path=self.data_ingestion_artifact.trained_file_path), DataValidation.read_file(file_path=self.data_ingestion_artifact.test_file_path))
            status=self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_err_msg+="Columns are missing in the train dataset"
            else:
                logging.info("All columns are there in the training dataset")
            status=self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_err_msg+="Columns are missing in the test dataset"
            else:
                logging.info("All columns are there in the testing dataset")
                
            status=self.is_column_exist(df=train_df)
            if not status:
                validation_err_msg+="Columns are missing in the training dataset"
            else:
                logging.info("All categorical/num columns are there in the training dataset")
            status=self.is_column_exist(df=test_df)
            if not status:
                validation_err_msg+="Columns are missing in the testimg dataset"
            else:
                logging.info("All categorical/num columns are there in the testing dataset")
             
            validation_status=len(validation_err_msg)==0
            
            data_validation_artifact = DataValidationArtifact(validation_status=validation_status, message=validation_err_msg, validation_report_file_path=self.data_validation_config.validation_report_file_path)
            
            report_dir=os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            
            validation_report={ "validation_status":validation_status, "message": validation_err_msg }
            
            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)
                
            logging.info("Data validation artifact created and saved to JSON file")
            logging.info("Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact            
            
        except Exception as e:
            raise MyException(e,sys)
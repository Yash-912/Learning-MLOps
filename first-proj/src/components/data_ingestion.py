import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
    def export_data_into_features(self) -> DataFrame:
        try:
            logging.info("Exporting data from mongoDB")
            mydata=Proj1Data()
            dataframe=mydata.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info("data loaded successfully")
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"saving export data into feature store file path {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_as_train_test(self,dataframe: DataFrame) -> None:
        logging.info("entered train test split")
        try:
            train_set, test_set=train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed train test split of the data ")
            logging.info("Exited train test split as method of data ingestion class")
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Exporting train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file path")
            
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe=self.export_data_into_features()
            logging.info("Got data from mongoDB")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exiting initiate_data_ing method of data_ingestion class")
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, 
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys) from e 
        
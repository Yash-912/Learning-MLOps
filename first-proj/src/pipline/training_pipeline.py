import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import ( DataIngestionConfig, DataValidationConfig )
from src.entity.artifact_entity import ( DataIngestionArtifact, DataValidationArtifact)

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion pipeline")
            logging.info("Getting data from mongodb")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info("Data gained from monoDB")
            logging.info("Exiting from data_ingestion method of TrainingPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("starting data validation")
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info("Performed data validation ")
            logging.info("Exiting from data_validation method of Training Pipeline")
            
            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
     
    def run_pipeline(self,) -> None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
        except Exception as e:
            raise MyException(e,sys)
        

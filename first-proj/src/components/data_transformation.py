import sys 
import os 
import numpy as np 
import pandas as pd 
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact,DataValidationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
    
    def get_transformer_object(self) -> Pipeline:
        logging.info("entered get_transformer_object method of datatransformation class")
        try:
            num_transformer=StandardScaler()
            min_Max_scaler=MinMaxScaler()
            logging.info("transformenrs initialized min-max and standard-scaler")
            
            num_features=self._schema_config['numerical_columns']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")
            
            preprocessor=ColumnTransformer(transformers=[
                ("StandardScaler", num_transformer, num_features),
                ("MinMaxScaler", min_Max_scaler, mm_columns)
            ], remainder='passthrough')
            
            final_pipeline=Pipeline(steps=[("preprocessor",preprocessor)])
            logging.info('Final pipeline ready')
            logging.info("Exiting get_transformer_object method of datatransformation class")
            return final_pipeline
        except Exception as e:
            logging.info("Exception occured in get_transformer_object of datatransformation")
            raise MyException(e,sys)
        
    def _map_gender_column(self, df):
        logging.info("mapping gender to numeric 0/1 values")
        df['Gender']=df['Gender'].map({'Female': 0,'Male': 1}).astype(int)
        return df
    
    def _create_dummy_columns(self,df):
        logging.info("creating dummy variables for cat columns")
        df=pd.get_dummies(df, drop_first=True)
        return df
    
    def _rename_cols(self, df):
        logging.info("Renaming some features and casting of the df")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df
    
    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df=self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df=self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Loaded test and train data")
            
            X_train=train_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_train=train_df[TARGET_COLUMN]
            X_test=test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test=test_df[TARGET_COLUMN]
            logging.info("X and y test and trained defined")
            
            X_train=self._map_gender_column(X_train)
            X_train=self._drop_id_column(X_train)
            X_train=self._create_dummy_columns(X_train)
            X_train=self._rename_cols(X_train)
            
            X_test=self._map_gender_column(X_test)
            X_test=self._drop_id_column(X_test)
            X_test=self._create_dummy_columns(X_test)
            X_test=self._rename_cols(X_test)
            logging.info("Transformations applied to X_train and X_test")
            
            logging.info("starting data transformation")
            preprocessor=self.get_transformer_object()
            logging.info("Got the preprocessor obj")
            
            logging.info("Initializing transformation for training data")
            X_train_arr=preprocessor.fit_transform(X_train)
            logging.info("initializing transformation for testing data")
            X_test_arr=preprocessor.transform(X_test)
            logging.info("transformation done completely for test and train")
            
            logging.info("Apply SMOTTEIN for handling imbalanced dataset")
            smt=SMOTEENN(sampling_strategy="minority")
            X_train_final, y_train_final = smt.fit_resample(X_train_arr, y_train)
            X_test_final  = X_test_arr
            y_test_final  = y_test
            logging.info("SMOTEENN spplied to train and test X,y ")
            
            train_arr=np.c_[X_train_final, np.array(y_train_final)]
            test_arr=np.c_[X_test_final, np.array(y_test_final)]
            logging.info("X and y joining done for train and test")
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            logging.info("Saved transformed object and files")
            
            logging.info("Data Transformation completed")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path
            )
        except Exception as e:
            raise MyException(e,sys)
import os 
import sys 
from typing import Tuple 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score, precision_score,recall_score 

from src.exception import MyException
from src.logger import logging 
from src.utils.main_utils import load_numpy_array_data,load_object,save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_trainer_config=model_trainer_config
        
    def get_model_object_and_report(self, train:np.array, test:np.array) -> Tuple[object,object]:
        try: 
            logging.info("Training RFC with given parameters")
            X_train, y_train, X_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info("train-test split done.")
            model = RandomForestClassifier(
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state
            )
            logging.info("Model training started")
            model.fit(X_train, y_train)
            logging.info("model training done ")
            
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred) 
            precision=precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            print(f"accuracy: {accuracy}")
            print(f"f1_score: {f1}")
            print(f"precision_score : {precision}")
            print(f"recall score : {recall}")
            
            metric_artifact=ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)     
            logging.info("Generated model metrics") 
            return model, metric_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("entered iterate_model_trainer method of model trainer class")
            logging.info('Starting model trainer component')
            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train and test data loaded")
            trained_model, metric_artifact=self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("trained model and metric loaded")
            preprocessor_obj=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessor obj loaded")
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")
            
            logging.info("saving this model with good performance ")
            my_model=MyModel(preprocessing_object=preprocessor_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            #save_object(file_path=self.model_trainer_config.metric_file_path, obj=metric_artifact) 

            logging.info("Saved the model with the trained model and preprocessor ")
         
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, metric_artifact=metric_artifact)
            logging.info(f"Model trainer artifact {model_trainer_artifact}")
            return model_trainer_artifact
        
        
        except Exception as e:
            raise MyException(e,sys)
     
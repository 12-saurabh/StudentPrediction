import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }
            
            
            params = {
                "DecisionTree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },

                "RandomForest": {
                    "n_estimators": [50, 100, 200]
                },

                "GradientBoosting": {
                    "learning_rate": [0.1, 0.01, 0.05],
                    "n_estimators": [50, 100]
                },

                "LinearRegression": {},

                "KNN": {
                    "n_neighbors": [5, 7, 9, 11]
                },

                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100]
                },

                "CatBoostRegressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.01],
                    "iterations": [50, 100]
                },

                "AdaBoostRegressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100]
                }
            }

            
            model_report,trained_models=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models,param=params)
            
            ##To get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            ##To get best model name from dict
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=trained_models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test,predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
        
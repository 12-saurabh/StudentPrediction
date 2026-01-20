import sys
import os
import numpy as np
import pandas as pd

import dill

from src.exception import CustomException

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            if model_name == "CatBoostRegressor":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                report[model_name] = score
                trained_models[model_name] = model
                continue

            gs = GridSearchCV(
                model,
                param[model_name],
                cv=3
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)

    
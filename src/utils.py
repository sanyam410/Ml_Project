import os
import sys
import pickle
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e  # type: ignore[arg-type]

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train, y_train)

            # Predict testing data and training data
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            # Get r2 score for the model
            test_model_score = r2_score(y_test, y_test_pred)
            train_model_score = r2_score(y_train, y_train_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys) from e  # type: ignore[arg-type]

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) #type: ignore
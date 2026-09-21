import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(".artifacts", "model.pkl")
class modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and testing input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models={
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "XGB Regressor": XGBRegressor(random_state=42, n_jobs=-1),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }

            parameter_distributions = {
                "Random Forest Regressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [1.0, "sqrt"]
                },
                "Decision Tree Regressor": {
                    "max_depth": [None, 3, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },
                "Gradient Boosting Regressor": {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "XGB Regressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [2, 3, 4, 5],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0],
                    "min_child_weight": [1, 3, 5]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }

            model_report = {}
            for model_name, model in models.items():
                if model_name in parameter_distributions:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=parameter_distributions[model_name],
                        n_iter=10,
                        cv=5,
                        scoring="r2",
                        random_state=42,
                        n_jobs=-1
                    )
                    search.fit(x_train, y_train)
                    models[model_name] = search.best_estimator_
                    model_report[model_name] = search.best_score_
                    logging.info(
                        f"Best parameters for {model_name}: {search.best_params_}"
                    )
                else:
                    model_report[model_name] = cross_val_score(
                        model, x_train, y_train, cv=5, scoring="r2"
                    ).mean()
                    model.fit(x_train, y_train)

            # To get the best model score from the dictionary
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found", sys) # type: ignore
            logging.info(
                f"Best model selected by cross-validation is {best_model_name} "
                f"with CV R2 score of {best_model_score}"
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys) from e  # type: ignore[arg-type]


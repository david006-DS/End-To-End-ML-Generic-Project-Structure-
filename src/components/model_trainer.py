import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f'Best found model on both training and testing dataset: {best_model}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            parameters = {
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error", "poisson"],
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "loss": ["squared_error", "huber"],
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100, 150],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, parameters
            )

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["test_score"]
            )

            best_model = None
            for model_name, model in models.items():
                param_grid = parameters.get(model_name, {})

                if param_grid:
                    grid_search = GridSearchCV(
                        model,
                        param_grid,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                    )
                    grid_search.fit(X_train, y_train)
                    fitted_model = grid_search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    fitted_model = model

                if model_name == best_model_name:
                    best_model = fitted_model
                    break

            best_model_score = model_report[best_model_name]["test_score"]
            best_params = model_report[best_model_name]["best_params"]

            print(f"Best model: {best_model_name}")
            print(f"Test R² score: {best_model_score:.4f}")
            print(f"Best parameters: {best_params}")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
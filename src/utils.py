import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    try:
        model_report = {}

        for model_name, model in models.items():
            param_grid = parameters.get(model_name, {})

            if param_grid:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                fitted_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                fitted_model = model.fit(X_train, y_train)
                best_params = {}

            y_train_pred = fitted_model.predict(X_train)
            y_test_pred = fitted_model.predict(X_test)

            model_report[model_name] = {
                "train_score": r2_score(y_train, y_train_pred),
                "test_score": r2_score(y_test, y_test_pred),
                "best_params": best_params
            }

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
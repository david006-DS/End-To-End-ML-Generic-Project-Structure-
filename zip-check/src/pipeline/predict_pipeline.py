import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class CustomData:
    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        return pd.DataFrame(
            [
                {
                    "gender": self.gender,
                    "race_ethnicity": self.race_ethnicity,
                    "parental_level_of_education": self.parental_level_of_education,
                    "lunch": self.lunch,
                    "test_preparation_course": self.test_preparation_course,
                    "reading_score": self.reading_score,
                    "writing_score": self.writing_score,
                }
            ]
        )


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("artifacts", "model.pkl")

    def predict(self, features):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            if not hasattr(model, "predict"):
                raise TypeError(f"Loaded model is not a valid estimator: {type(model)}")

            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model         = load_object(os.path.join("artifacts", "model.pkl"))
        self.encoder_top10 = load_object(os.path.join("artifacts", "encoder_top10.pkl"))
        self.label_encoder = load_object(os.path.join("artifacts", "label_encoder.pkl"))
        self.top_features  = load_object(os.path.join("artifacts", "top_features.pkl"))
        logging.info("All artifacts loaded successfully")

    def predict(self, input_df: pd.DataFrame) -> list:
        try:
            logging.info("Starting prediction")

            missing = set(self.top_features) - set(input_df.columns)
            if missing:
                raise CustomException(f"Input is missing required columns: {missing}", sys)

            input_df      = input_df[self.top_features]
            encoded       = self.encoder_top10.transform(input_df)
            y_pred_enc    = self.model.predict(encoded)
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_enc)

            logging.info(f"Prediction complete: {y_pred_labels}")
            return y_pred_labels.tolist()

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Defect_of_vehicle: str,
        Age_band_of_driver: str,
        Road_surface_type: str,
        Vehicle_driver_relation: str,
        Weather_conditions: str,
        Types_of_Junction: str,
        Light_conditions: str,
        Day_of_week: str,
        Number_of_casualties: int,
        Number_of_vehicles_involved: int,
    ):
        self.Defect_of_vehicle           = Defect_of_vehicle
        self.Age_band_of_driver          = Age_band_of_driver
        self.Road_surface_type           = Road_surface_type
        self.Vehicle_driver_relation     = Vehicle_driver_relation
        self.Weather_conditions          = Weather_conditions
        self.Types_of_Junction           = Types_of_Junction
        self.Light_conditions            = Light_conditions
        self.Day_of_week                 = Day_of_week
        self.Number_of_casualties        = Number_of_casualties
        self.Number_of_vehicles_involved = Number_of_vehicles_involved

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Defect_of_vehicle":           [self.Defect_of_vehicle],
                "Age_band_of_driver":          [self.Age_band_of_driver],
                "Road_surface_type":           [self.Road_surface_type],
                "Vehicle_driver_relation":     [self.Vehicle_driver_relation],
                "Weather_conditions":          [self.Weather_conditions],
                "Types_of_Junction":           [self.Types_of_Junction],
                "Light_conditions":            [self.Light_conditions],
                "Day_of_week":                 [self.Day_of_week],
                "Number_of_casualties":        [self.Number_of_casualties],
                "Number_of_vehicles_involved": [self.Number_of_vehicles_involved],
            }
            df = pd.DataFrame(data)
            logging.info(f"Input DataFrame:\n{df}")
            return df

        except Exception as e:
            raise CustomException(e, sys)
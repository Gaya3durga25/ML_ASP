import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ===========================
    # TIME FEATURE ENGINEERING
    # ===========================
    def preprocess_time(self, df):
        try:
            logging.info("Starting time feature extraction")

            df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors='coerce')
            df["hour"]   = df["Time"].dt.hour
            df["minute"] = df["Time"].dt.minute
            df.drop(columns=["Time"], inplace=True)

            logging.info("Time feature extraction completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ===========================
    # PREPROCESSOR (ENCODING) — features only, NO target
    # ===========================
    def get_data_transformer_object(self, feature_columns):
        """
        Builds an OrdinalEncoder pipeline for the given feature columns.
        Target column is intentionally excluded here — handled in ModelTrainer.
        """
        try:
            logging.info("Creating preprocessing object")

            cat_pipeline = Pipeline(steps=[
                ("encoder", OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ])

            preprocessor = ColumnTransformer([
                ("cat_pipeline", cat_pipeline, feature_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ===========================
    # MAIN FUNCTION
    # ===========================
    def initiate_data_transformation(self, train_path, test_path):
        """
        Returns:
            train_arr        : encoded features + raw target (numpy array)
            test_arr         : encoded features + raw target (numpy array)
            train_df         : raw DataFrame (used by ModelTrainer for top-10 selection)
            test_df          : raw DataFrame (used by ModelTrainer for top-10 selection)
        
        NOTE: target column is kept raw (string) in train_arr/test_arr last column.
              ModelTrainer handles LabelEncoding of target once, cleanly.
        """
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            # Apply time feature engineering
            train_df = self.preprocess_time(train_df)
            test_df  = self.preprocess_time(test_df)

            logging.info("Time transformation applied")

            target_column = "Accident_severity"

            # Separate features and target
            input_train = train_df.drop(columns=[target_column])
            input_test  = test_df.drop(columns=[target_column])

            target_train = train_df[target_column].values  # raw strings, NOT encoded here
            target_test  = test_df[target_column].values

            # Build and fit preprocessor on feature columns only
            feature_columns    = input_train.columns.tolist()
            preprocessing_obj  = self.get_data_transformer_object(feature_columns)

            input_train_arr = preprocessing_obj.fit_transform(input_train)
            input_test_arr  = preprocessing_obj.transform(input_test)

            # Combine encoded features + raw target into arrays
            # ModelTrainer will slice off the last column for target encoding
            train_arr = np.c_[input_train_arr, target_train]
            test_arr  = np.c_[input_test_arr,  target_test]

            # Save full-feature preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")

            # Return raw DataFrames too — ModelTrainer needs column names for feature selection
            return train_arr, test_arr, train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)


# ===========================
# MAIN BLOCK
# ===========================
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        print("Data Ingestion Completed")
        print("Train Path:", train_path)
        print("Test Path:", test_path)

        transformation = DataTransformation()

        # Correctly unpacks all 4 return values
        train_arr, test_arr, train_df, test_df = transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        print("Data Transformation Completed")
        print("Train Array Shape:", train_arr.shape)
        print("Test Array Shape:",  test_arr.shape)
        print("Train DF Shape:",    train_df.shape)
        print("Test DF Shape:",     test_df.shape)

    except Exception as e:
        raise CustomException(e, sys)
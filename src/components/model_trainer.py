import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:   str = os.path.join("artifacts", "model.pkl")
    selected_features_file_path: str = os.path.join("artifacts", "top_features.pkl")
    label_encoder_path:        str = os.path.join("artifacts", "label_encoder.pkl")
    encoder_top10_path:        str = os.path.join("artifacts", "encoder_top10.pkl")  # NEW


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, train_df, test_df):
        try:
            logging.info("Starting model training")

            target_column = "Accident_severity"

            # STEP 1: Encode target ONCE here (raw strings from train_arr last col)
            label_encoder = LabelEncoder()

            y_train_raw = train_arr[:, -1]   # raw strings e.g. "Slight", "Serious"
            y_test_raw  = test_arr[:, -1]

            y_train = label_encoder.fit_transform(y_train_raw)
            y_test  = label_encoder.transform(y_test_raw)

            # Encoded features from DataTransformation
            X_train_all = train_arr[:, :-1].astype(float)
            X_test_all  = test_arr[:, :-1].astype(float)

           
            # STEP 2: Train on ALL features — just to get feature importance
           
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "ExtraTrees":   ExtraTreesClassifier(n_estimators=100,  random_state=42),
                "XGBoost":      XGBClassifier(eval_metric='mlogloss', random_state=42)
            }

            model_scores = {}

            for name, model in models.items():
                model.fit(X_train_all, y_train)
                y_pred = model.predict(X_test_all)
                score  = accuracy_score(y_test, y_pred)
                model_scores[name] = score
                logging.info(f"{name} Accuracy (all features): {score:.4f}")
                print(f"{name} Accuracy: {score:.4f}")

           
            # STEP 3: Select best model + top 10 features
         
            best_model_name = max(model_scores, key=model_scores.get)
            best_model      = models[best_model_name]

            if model_scores[best_model_name] < 0.5:
                raise CustomException("Best model accuracy too low to proceed", sys)

            print(f"\nBest Model: {best_model_name} ({model_scores[best_model_name]:.4f})")
            logging.info(f"Best model: {best_model_name}")

            importances  = best_model.feature_importances_
            feature_names = train_df.drop(columns=[target_column]).columns.tolist()

            top_indices  = np.argsort(importances)[-10:]          # indices of top 10
            top_features = [feature_names[i] for i in top_indices]

            print(f"Top 10 Features: {top_features}")
            logging.info(f"Top 10 features: {top_features}")

           
            # STEP 4: Build encoder_top10 from RAW top-10 columns
        
            # Use raw DataFrames (not encoded arrays) so the encoder
            # sees original values — exactly what prediction will receive
            X_train_top_raw = train_df[top_features]
            X_test_top_raw  = test_df[top_features]

            top10_pipeline = Pipeline(steps=[
                ("encoder", OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ])

            encoder_top10 = ColumnTransformer([
                ("top10_encoder", top10_pipeline, top_features)
            ])

            X_train_top = encoder_top10.fit_transform(X_train_top_raw)
            X_test_top  = encoder_top10.transform(X_test_top_raw)

           
            # STEP 5: Retrain best model on encoded top 10 only
           
            best_model.fit(X_train_top, y_train)

            y_pred_final   = best_model.predict(X_test_top)
            final_accuracy = accuracy_score(y_test, y_pred_final)

            print(f"\nFinal Accuracy (top 10 features): {final_accuracy:.4f}")
            logging.info(f"Final accuracy: {final_accuracy:.4f}")

            # ===========================
            # STEP 6: Save all 4 artifacts
            # ===========================
            save_object(self.config.trained_model_file_path,     best_model)
            save_object(self.config.selected_features_file_path, top_features)
            save_object(self.config.label_encoder_path,          label_encoder)
            save_object(self.config.encoder_top10_path,          encoder_top10)  # NEW

            logging.info("All artifacts saved successfully")
            logging.info("Model training completed")

            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)


# ===========================
# MAIN BLOCK
# ===========================
if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion
        from src.components.data_transformation import DataTransformation

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, train_df, test_df = transformation.initiate_data_transformation(
            train_path,
            test_path
        )

        # Step 3: Model Training
        trainer  = ModelTrainer()
        accuracy = trainer.initiate_model_trainer(
            train_arr, test_arr,
            train_df,  test_df
        )

        print(f"\nPipeline Complete. Final Model Accuracy: {accuracy:.4f}")
        print("\nSaved artifacts:")
        print("  artifacts/model.pkl")
        print("  artifacts/top_features.pkl")
        print("  artifacts/label_encoder.pkl")
        print("  artifacts/encoder_top10.pkl")

    except Exception as e:
        raise CustomException(e, sys)
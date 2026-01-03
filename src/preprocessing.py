# importing libraries
import os
import sys
import pandas as pd 
import numpy as np 

#importing dependencies
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.utils.common import save_object
from src.config.paths import PROCESSED_DATA_DIR, ARTIFACT_DIR

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, target_column : str = "Churn"):
        self.target_column= target_column

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Data cleaning Initiated")

            #Dropping unwanted column
            drop_list = ["customerID", "gender"]
            df = df.drop(columns =drop_list, errors="ignore")
            
            # Removing null values from TotalCharges
            #First of all in order to drop " " , we need to replace it with np.nan
            df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
            df = df.dropna(subset=["TotalCharges"])
            df["TotalCharges"] = df["TotalCharges"].astype(float)

            logger.info("Data cleaning completed.")
            return df

        except Exception as e:
            logger.exception("Failed to clean Data!")
            raise ChurnException(e, sys)

    def get_preprocessor(self, df:pd.DataFrame)->ColumnTransformer:
        try:
            logger.info("Creating Preprocessing pipeline.")

            categorical_column =df.select_dtypes(include="object").columns
            numerical_column = df.select_dtypes(exclude="object").columns

            logger.info(f"Categorical Column : {list(categorical_column)}")
            logger.info(f"Numerical Column : {list(numerical_column)}")

            preprocessor = ColumnTransformer(
                [
                    ("scaler", StandardScaler(), numerical_column),
                    ("oh_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_column)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logger.exception("Failed to preprocess data!")
            raise ChurnException(e, sys)

    def initiate_preprocessing(self):
        try:
            logger.info("Preprocessing Started")

            train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
            test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

            #loading dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #Cleaning data
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            X_train = train_df.drop(columns=self.target_column)
            y_train = train_df[self.target_column]

            X_test = test_df.drop(columns=self.target_column)
            y_test = test_df[self.target_column]

            preprocessor = self.get_preprocessor(X_train)

            logger.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logger.info("Fitting preprocessor on testing data")
            X_test_transformed = preprocessor.transform(X_test)

            preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
            save_object(preprocessor_path, preprocessor)

            logger.info("Preprocessing completed successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                y_test.values,
                preprocessor_path
            )
            
        except Exception as e:
            logger.exception("Failed to preprocessing!")
            raise ChurnException(e, sys)

if __name__ == "__main__":
    obj = DataPreprocessing()
    x_1, x_2, y_1, y_2 ,path= obj.initiate_preprocessing()
    print(f"🦜testing Data : {x_2}")
    print(f"🦚testing data y_values : {y_2}")
    print(f"💃Preprocessed file path : {path}")
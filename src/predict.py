#Importing library
import os
import sys
import pandas as pd
import numpy as np

#importing dependencies
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.utils.common import load_object
from src.config.paths import ARTIFACT_DIR, RAW_DATA_DIR

logger = get_logger(__name__)

class ChurnPredictor:
    def __init__(self, threshold: float =0.35):   
        self.threshold = threshold
        try:
            logger.info("Loading prediction artifacts")
            preprocessor_path = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
            model_path = os.path.join(ARTIFACT_DIR, "Churn_Model.pkl")
            le_churn_path  = os.path.join(ARTIFACT_DIR, "le_churn.pkl")

            self.preprocessor= load_object(preprocessor_path)
            self.model = load_object(model_path)
            self.le_churn = load_object(le_churn_path)

            logger.info("All artifacts loaded succesfully.")

        except Exception as e :
            logger.exception("Failed to load prediction artifacts.")
            raise ChurnException(e, sys)

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try: 
            logger.info("Starting prediction Successfully.")

            #transform the input
            X_preprocessed= self.preprocessor.transform(input_df)

            #predict probability
            prob = self.model.predict_proba(X_preprocessed)[:, 1]

            #Applying threshold
            churn_pred_numeric = (prob >=self.threshold).astype(int)

            #Convert numeric prediction back to label
            churn_pred_label = self.le_churn.inverse_transform(churn_pred_numeric)

            #Final output
            result = input_df.copy()
            result["churn_probability"] = prob
            result["churn_prediction"] = churn_pred_label

            logger.info("Prediction Completed successfully.")
            return result

        except Exception as e:
            logger.exception("Failed to predict output.")
            raise ChurnException(e, sys)

if __name__ =="__main__":

    #Example
    raw_data_path= os.path.join(RAW_DATA_DIR, "Telco_customer_churn.csv")
    sample_df = pd.read_csv(raw_data_path).sample(n=5, random_state=42)

    #drop churn column 
    if "Churn" in sample_df.columns:
        true_label = sample_df["Churn"]
        sample_df = sample_df.drop(columns='Churn')
        logger.info(f"true label for sample dataset is as {true_label}")

    predictor =ChurnPredictor()
    prediction = predictor.predict(sample_df)
    print(prediction)

    

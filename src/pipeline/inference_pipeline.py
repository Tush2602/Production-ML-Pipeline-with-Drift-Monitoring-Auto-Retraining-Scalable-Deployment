import os
import sys
import pandas as pd 

from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.predict import ChurnPredictor

logger = get_logger(__name__)

class InferencePipeline:
    def __init__(self, threshold : float= 0.35):
        try:
            logger.info("Initiated Inference pipeline.")
            self.predictor =ChurnPredictor(threshold = threshold)
            logger.info("Inference Pipeline initialized successfully.")

        except Exception as e: 
            logger.exception("Failed to initialize inference pipeline!")
            raise ChurnException(e, sys)

    def predict(self, input_data):
        try:
            logger.info("Starting Interference.")

            #convert input to dataframe if required
            if isinstance(input_data, dict):
                input_df =pd.DataFrame([input_data])

            elif isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)

            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data

            else:
                raise ValueError("Invalid input type for predictions.")

            logger.info(f"Inference input shape: {input_df.shape}")

            result = self.predictor.predict(input_df)

            logger.info("Inference completed successfully")
            return result

        except Exception as e:
            logger.exception("Inference failed")
            raise ChurnException(e, sys)

if __name__ == "__main__":
    #Example 
    sample_input = {
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.35,
        "TotalCharges": 904.2
    }

    pipeline = InferencePipeline()
    output = pipeline.predict(sample_input)
    print(output)

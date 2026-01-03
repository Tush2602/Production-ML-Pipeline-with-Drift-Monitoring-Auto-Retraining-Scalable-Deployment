#importing libraries
import os
import sys
import pandas as pd 
import numpy as np 

#importing dependencies
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.utils.common import save_object
from src.config.paths import ARTIFACT_DIR
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import DataPreprocessing

#importing models
from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(
            C=100,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs"
        )

    def initiate_training(self, X_train : np.ndarray, y_train: np.ndarray):
        try:
            logger.info("Model Trainig started.")

            #Encoding Target variable
            le_churn = LabelEncoder()
            y_train_encoded = le_churn.fit_transform(y_train)

            logger.info("Target variable encoded successfully.")

            #Training Model
            self.model.fit(X_train, y_train_encoded)

            logger.info("Model Training Completed")

            #Saving model to local
            model_path = os.path.join(ARTIFACT_DIR, "Churn_Model.pkl")
            save_object(model_path, self.model)

            #Saving label encoder instance to local
            le_path = os.path.join(ARTIFACT_DIR, "le_churn.pkl")
            save_object(le_path, le_churn)

            logger.info("Model and Label encoder instance saved to local successfully.")

            return model_path, le_path


        except Exception as e:
            logger.exception("Model Training Failed")
            raise ChurnException(e, sys)


if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    X_train, X_test, y_train, y_test, _ = preprocessor.initiate_preprocessing()

    trainer =ModelTrainer()
    model_path, encoder_path = trainer.initiate_training(X_train, y_train)
    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Label encoder saved at: {encoder_path}")
    
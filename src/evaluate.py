#importing library
import os 
import sys
import pandas as pd
import numpy as np 
import json

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.utils.common import load_object, eval_metrics, save_json
from src.config.paths import ARTIFACT_DIR
from src.preprocessing import DataPreprocessing

logger = get_logger(__name__)

class ModelEval:
    def __init__(self, threshold : float):
        self.threshold = threshold

    def initiate_eval(self, X_test : np.ndarray, y_test : np.ndarray):
        try:
            logger.info("Model Evaluation started.")

            #Load trained model
            model_path = os.path.join(ARTIFACT_DIR, "Churn_Model.pkl")
            model = load_object(model_path)

            #Prediction of probablity
            y_test_prob = model.predict_proba(X_test)[:, 1]

            #Apply Custom threshold 
            y_pred = (y_test_prob >=self.threshold).astype(int)

            #Apply encoding to y_test
            le_churn_path = os.path.join(ARTIFACT_DIR, "le_churn.pkl")
            le_churn = load_object(le_churn_path)

            y_test_encoded = le_churn.transform(y_test)

            #Evaluating using Metrics 
            acc, recall, f1, prec, roc_auc, report = eval_metrics(y_test_encoded, y_pred, y_test_prob)

            #Saving metrics to artifacts
            metrics = {
                "Threshold" :self.threshold,
                "accuracy score": acc,
                "recall score" : recall,
                "f1 score" : f1,
                "Precision score" : prec,
                "ROC-AUC score" : roc_auc,
                "Classification Report" : report
            }

            metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
            save_json(metrics_path, metrics)

            logger.info("Model evaluation completed successfully")
            logger.info(f"Evaluation metrics: {metrics}")

            production_metrics_path = os.path.join(ARTIFACT_DIR, "production_metrics.json")

            # If no production model exists, promote automatically
            if not os.path.exists(production_metrics_path):
                save_json(production_metrics_path, metrics)
                logger.info("No production model found. Promoting candidate as production.")
                return metrics_path, metrics

            # Load existing production metrics
            with open(production_metrics_path, "r") as f:
                production_metrics = json.load(f)

            # Compare models
            if is_model_better(metrics, production_metrics):
                save_json(production_metrics_path, metrics)
                logger.info("Candidate model promoted to production.")
            else:
                logger.warning("Candidate model rejected. Production model retained.")

            return metrics_path, metrics

        except Exception as e :
            logger.exception("Failed to evaluate model !")
            raise ChurnException(e, sys)


def is_model_better(candidate_metrics: dict, production_metrics: dict) -> bool:
    return (
        candidate_metrics["ROC-AUC score"] >= production_metrics["ROC-AUC score"]
        and
        candidate_metrics["recall score"] >= production_metrics["recall score"]
    )


if __name__ == "__main__":
    preprocessor = DataPreprocessing()
    X_train, X_test, y_train, y_test, _ = preprocessor.initiate_preprocessing()

    evaluator = ModelEval(threshold=0.35)
    path, metric = evaluator.initiate_eval(X_test, y_test)

    print(f"Metrics is save at path : {path}")
    print(metric)

        
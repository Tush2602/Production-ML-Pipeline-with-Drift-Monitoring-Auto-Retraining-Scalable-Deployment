#importing library
import sys
import os
import json
import numpy as np
import pandas as pd

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.utils.common import load_object, save_json

from src.monitoring.drift_detection import DataDriftDetector
from src.pipeline.training_pipeline import TrainingPipeline
from src.config.paths import PROCESSED_DATA_DIR, DRIFT_DATA_DIR, ARTIFACT_DIR


logger = get_logger(__name__)

class RetrainingTrigger:
    def __init__(self):
        self.drift_detector = DataDriftDetector()

    def run(self):
        try:
            logger.info("Starting retraining trigger check")

            # 1. Load reference (training Data)
            reference_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
            reference_df=pd.read_csv(reference_path)

            #drop target column if present
            if "Churn" in reference_df.columns:
                reference_df = reference_df.drop(columns = ["Churn"])

            # 2. Load current data
            current_path = os.path.join(DRIFT_DATA_DIR, "current_data.csv")

            if not os.path.exists(current_path):
                logger.warning(
                    "No current inference data found. Skipping drift check."
                )
                return

            current_df = pd.read_csv(current_path)
            if current_df.empty:
                logger.warning("Current inference data is empty. Skipping retraining.")
                return

            # 3. Detect Drift
            drift_result = self.drift_detector.detect_drift(
                reference_df, current_df
            )

            confidence_degraded = is_confidence_degraded(current_df)

            if drift_result["drift_detected"]:
                logger.warning("Data Drift Detected!")

            if confidence_degraded:
                logger.warning("Prediction confidence degraded!")

            # 4. Trigger retraining if drift detected
            if drift_result["drift_detected"] or confidence_degraded:
                logger.warning("Triggered Retraining.")

                training_pipeline =TrainingPipeline()
                training_pipeline.run()

                logger.info("✅ Model retraining completed successfully")

            else:
                logger.info("✅ No retraining conditions met. System healthy.")

        except Exception as e:
            logger.exception("Retraining trigger failed")
            raise ChurnException(e, sys)


def compute_mean_confidence(probabilities: np.ndarray) -> float:
    return float(np.mean(np.abs(probabilities - 0.5)))


def is_confidence_degraded(current_df: pd.DataFrame, threshold: float = 0.15) -> bool:
    confidence_path = os.path.join(ARTIFACT_DIR, "confidence_baseline.json")

    # No baseline yet → cannot judge degradation
    if not os.path.exists(confidence_path):
        logger.info("No confidence baseline found. Skipping confidence check.")
        return False

    # Load baseline
    with open(confidence_path, "r") as f:
        baseline = json.load(f)

    # Load model & preprocessor
    model = load_object(os.path.join(ARTIFACT_DIR, "Churn_Model.pkl"))
    preprocessor = load_object(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))

    # Prepare data
    X_current = preprocessor.transform(current_df)

    # Predict probabilities
    probs = model.predict_proba(X_current)[:, 1]

    current_confidence = compute_mean_confidence(probs)
    baseline_confidence = baseline["mean_confidence"]

    logger.info(
        f"Current confidence={current_confidence:.4f}, "
        f"Baseline confidence={baseline_confidence:.4f}"
    )
    return current_confidence < (baseline_confidence * (1 - threshold))



if __name__ == "__main__":
    retraining = RetrainingTrigger()
    retraining.run()


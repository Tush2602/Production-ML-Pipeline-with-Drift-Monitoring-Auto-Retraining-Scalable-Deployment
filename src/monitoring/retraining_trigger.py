#importing library
import sys
import os
import pandas as pd

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException

from src.monitoring.drift_detection import DataDriftDetector
from src.pipeline.training_pipeline import TrainingPipeline
from src.config.paths import PROCESSED_DATA_DIR, DRIFT_DATA_DIR

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

            # 4. Trigger retraining if drift detected
            if drift_result["drift_detected"]:
                logger.warning("Data Drift Detected! Triggered Retraining.")

                training_pipeline =TrainingPipeline()
                training_pipeline.run()

                logger.info("✅ Model retraining completed successfully")

            else:
                logger.info("✅ No significant drift detected. Retraining not required.")

        except Exception as e:
            logger.exception("Retraining trigger failed")
            raise ChurnException(e, sys)


if __name__ == "__main__":
    retraining = RetrainingTrigger()
    retraining.run()


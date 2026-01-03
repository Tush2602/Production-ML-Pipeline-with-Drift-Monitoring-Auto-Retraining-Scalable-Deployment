#importing library
import os
import sys

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException

from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessing
from src.train import ModelTrainer
from src.evaluate import ModelEval

logger =get_logger(__name__)

class TrainingPipeline:
    def run(self):
        try:
            logger.info("Training Pipeline initiated.")

            # 1. DataIngestion
            ingestion =DataIngestion()
            train_path, test_path =ingestion.initiate_data_ingestion()

            logger.info(f"Train Data path : {train_path}")
            logger.info(f"Test file path : {test_path}")

            #2. Data Preprocessing
            preprocessor = DataPreprocessing()
            X_train, X_test, y_train, y_test , preprocessor_path = preprocessor.initiate_preprocessing()

            logger.info(f"preprocessor object save at {preprocessor_path}")

            #3. Model training
            trainer = ModelTrainer()
            model_path, le_path = trainer.initiate_training(X_train, y_train)

            logger.info(f"Model saved at {model_path}")
            logger.info(f"label encoder saved at path {le_path}")

            # 4. Model evaluation
            evaluator = ModelEval(threshold=0.35)
            metrics_path , metrics = evaluator.initiate_eval(X_test, y_test)

            logger.info(f"Metrics saved at: {metrics_path}")
            logger.info(f"Evaluation metrics: {metrics}")

            logger.info("="* 85)
            logger.info(f"Training pipeline completed successfully.")
            logger.info("="* 85)

            return metrics

        except Exception as e:
            logger.exception("Failed to run pipeline!")
            raise ChurnException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
#Importing libraries
import os 
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException
from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR



logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, test_size:float=0.2, random_state = 42):
        self.test_size= test_size
        self.random_state = random_state

    def initiate_data_ingestion(self):
        try:
            logger.info("Data Ingestion Started.")
            raw_file_path= os.path.join(RAW_DATA_DIR, "telco_customer_churn.csv")
            
            if not os.path.exists(raw_file_path):
                raise FileNotFoundError(f"Raw Data was not found at {raw_file_path}")

            logger.info(f"Reading raw data file from {raw_file_path}")
            df = pd.read_csv(raw_file_path)

            logger.info(f"Splitting data into train and test sets.")
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state, stratify=df['Churn'])

            os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

            train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
            test_path  = os.path.join(PROCESSED_DATA_DIR, "test.csv")

            train_df.to_csv(train_path, index=False)
            logger.info(f"Training data saved successfully saved at {train_path}")

            test_df.to_csv(test_path, index= False)
            logger.info(f"Testing data saved successfully saved at {test_path}")

            logger.info("Data Ingestion Completed successfully")

            return train_path , test_path
        except Exception as e:
            logger.exception("Data Ingestion Failed.")
            raise ChurnException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    training, testing = obj.initiate_data_ingestion()
    print(f"Training data file is {training} and testing file is {testing}.")

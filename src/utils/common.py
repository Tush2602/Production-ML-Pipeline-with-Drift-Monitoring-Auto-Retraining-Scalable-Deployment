#Importing library 
import os
import sys
import pickle
import json


from src.utils.exception import ChurnException
from src.utils.logger import get_logger

from sklearn.metrics import (accuracy_score, 
                            recall_score, 
                            f1_score, 
                            precision_score, 
                            roc_auc_score, 
                            classification_report,
                            )

logger = get_logger(__name__)

#Creating a function for saving object
def save_object(file_path:str, obj:object):
    try: 
        # Creating directory 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info("Object saved successfully")
    except Exception as e:
        logger.exception("Error Occured while saving object")
        raise ChurnException(e, sys)

#Creating a function to load object
def load_object(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found at {file_path}")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logger.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        logger.exception("Error Occured while loading object")
        raise ChurnException(e, sys)

# Creating a function for metrics evaluation and Roc-auc curve plotting

def eval_metrics(true, pred, prob):
    try:
        acc = accuracy_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
        prec = precision_score(true, pred)
        roc_auc = roc_auc_score(true, prob)
        report= classification_report(true, pred, output_dict=True)

        logger.info("Metrics Evaluated successfully.")
        return acc, recall, f1, prec, roc_auc, report

    except Exception as e:
        logger.exception("Failed to evaluate metrics!")
        raise ChurnException(e, sys)


#Creating a function to save json file 
def save_json(file_path : str, json_obj):
    try:
        #Creating Directory
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(json_obj, f, indent =4)

        logger.info("JSON Object saved successfully.")

    except Exception as e:
        logger.exception("Error Occured while saving json files.")
        raise ChurnException(e, sys)

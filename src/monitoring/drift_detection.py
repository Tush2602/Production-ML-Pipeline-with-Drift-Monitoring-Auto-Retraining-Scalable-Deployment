#importing library 
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict

#importing requirements
from src.utils.logger import get_logger
from src.utils.exception import ChurnException 

logger = get_logger(__name__)

class DataDriftDetector:
    def __init__(self, p_value_threshold:float = 0.05, drift_ratio_threshold:float = 0.3):
        self.p_value_threshold = p_value_threshold
        self.drift_ratio_threshold = drift_ratio_threshold

    def detect_drift(self, reference_df :pd.DataFrame, current_df : pd.DataFrame) -> Dict:
        try:
            logger.info("Starting Data drift detection.")

            drifted_features =[]

            #select only numeric columns
            numerical_column =reference_df.select_dtypes(exclude="object").columns

            logger.info(f"Numerical columns Checked : {list(numerical_column)}")

            for col in numerical_column:
                ref_data = reference_df[col].dropna()
                cur_data = current_df[col].dropna()

                if len(ref_data) ==0 or len(cur_data) ==0 :
                    continue

                _, p_value= ks_2samp(ref_data, cur_data)

                if p_value < self.p_value_threshold:
                    drifted_features.append(col)
                    logger.warning(f"Drift Detected in feature {col} (p-value={p_value})")

            drift_ratio = (
                len(drifted_features) / len(numerical_column) if len(numerical_column) > 0 else 0
            )


            drift_detected = drift_ratio >= self.drift_ratio_threshold

            result = {
                "drift_detected": drift_detected,
                "drift_ratio": drift_ratio,
                "drifted_features": drifted_features
            }

            logger.info(f"Drift detection result: {result}")
            return result
        
        except Exception as e:
            logger.exception('Drift Detection Failed.')
            raise ChurnException(e, sys)

import os

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#logs directory
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#config directory
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

#data directory
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DRIFT_DATA_DIR = os.path.join(DATA_DIR, "drift")

#artifact directory
ARTIFACT_DIR = os.path.join(DATA_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

#model directory
MODEL_DIR = os.path.join(ARTIFACT_DIR, "model")





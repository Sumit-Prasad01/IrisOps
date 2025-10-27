import os

# Data Ingestion
RAW_DIR = 'artifacts/raw'
RAW_FILE_PATH = os.path.join(RAW_DIR, "data.csv")
DATA_DIR = "artifacts/raw/data.csv"

# Data Processing
PROCESSED_DIR = "artifacts/processed"

# Features Path
## Save 
X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.pkl")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.pkl")
y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.pkl")
y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.pkl")

# load path
X_TRAIN_LOAD_PATH = 'artifacts/processed/X_train.pkl'
X_TEST_LOAD_PATH = 'artifacts/processed/X_test.pkl'
y_TRAIN_LOAD_PATH = 'artifacts/processed/y_train.pkl'
y_TEST_LOAD_PATH = 'artifacts/processed/y_test.pkl'

# Model Training
MODEL_PATH = "artifacts/models"
SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "dt_model.pkl")
SAVED_MODEL_PATH = "artifacts/models/dt_model.pkl"

# Visuals Path
VISUALS_PATH = "artifacts/visuals"
os.makedirs(VISUALS_PATH, exist_ok = True)
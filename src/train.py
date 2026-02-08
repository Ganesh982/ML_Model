import os
import logging
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = r"C:\Users\Thulasi\Desktop\ML_Model\data\loan_default_dataset.csv"
MODEL_PATH = r"C:\Users\Thulasi\Desktop\ML_Model\models/loan_default_model.pkl"

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO)
logging.info("Training started")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("default", axis=1)
y = df["default"]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# PIPELINE
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# -------------------------
# TRAIN MODEL
# -------------------------
pipeline.fit(X_train, y_train)
logging.info("Model training completed")

# -------------------------
# CREATE MODELS FOLDER
# -------------------------
os.makedirs("models", exist_ok=True)

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(pipeline, MODEL_PATH)
logging.info(f"Model saved at {MODEL_PATH}")

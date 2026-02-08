import joblib
import pandas as pd
from config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(data: dict):
    df = pd.DataFrame([data])
    return model.predict(df)[0]

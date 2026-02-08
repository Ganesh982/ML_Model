from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

# -------------------------
# CREATE APP
# -------------------------
app = FastAPI()

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("models/loan_default_model.pkl")
print("Model loaded successfully")

# -------------------------
# HOME ENDPOINT
# -------------------------
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}

# -------------------------
# SINGLE PREDICTION
# -------------------------
@app.post("/predict")
def predict_default(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    print("Input:", data)
    print("Prediction:", prediction)

    return {"default_prediction": int(prediction)}

# -------------------------
# FILE UPLOAD PREDICTION
# -------------------------
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Drop target column if present
        if "default" in df.columns:
            df = df.drop("default", axis=1)

        predictions = model.predict(df)
        df["prediction"] = predictions

        return df.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}


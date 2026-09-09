from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import joblib
import pandas as pd

pipeline = joblib.load("heart_pipeline.pkl")

app = FastAPI(title="Heart Disease Prediction API")


class UserInput(BaseModel):
    Age: Annotated[int, Field(..., gt=0, le=120, description="Age of the patient")]
    Sex: Annotated[Literal["M", "F"], Field(..., description="Sex of the patient")]
    ChestPainType: Annotated[
        Literal["ATA", "NAP", "ASY", "TA"], Field(..., description="Chest pain type")
    ]
    RestingBP: Annotated[
        int, Field(..., gt=30, lt=370, description="Resting blood pressure (mm Hg)")
    ]
    Cholesterol: Annotated[
        int, Field(..., ge=0, lt=1000, description="Serum cholesterol (mm/dl)")
    ]
    FastingBS: Annotated[
        Literal[0, 1], Field(..., description="1 if fasting blood sugar > 120 mg/dl else 0")
    ]
    RestingECG: Annotated[
        Literal["Normal", "ST", "LVH"], Field(..., description="Resting ECG result")
    ]
    MaxHR: Annotated[int, Field(..., gt=0, description="Maximum heart rate achieved")]
    ExerciseAngina: Annotated[
        Literal["Y", "N"], Field(..., description="Exercise-induced angina")
    ]
    Oldpeak: Annotated[
        float, Field(..., ge=0, le=6, description="ST depression induced by exercise")
    ]
    ST_Slope: Annotated[
        Literal["Up", "Flat", "Down"], Field(..., description="Slope of peak exercise ST segment")
    ]


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running. POST to /predict."}


@app.post("/predict")
def predict(data: UserInput):
    
    input_df = pd.DataFrame([data.model_dump()])

    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "result": "Heart Disease Likely" if prediction == 1 else "No Heart Disease",
        "probability_of_heart_disease": round(probability, 4),
    }
"""
API census
Author: Sunil Mudumala
Date: 23/Sep/2023
"""

import uvicorn
import pandas as pd
import pickle
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from ml.data import process_data

MODEL_PATH = Path(__name__).resolve().parent / "model"
ENCODER_FILE = MODEL_PATH / "encoder.pkl"
LB_FILE = MODEL_PATH / "label_binarizer.pkl"
MODEL_FILE = MODEL_PATH / "trained_model.pkl"


class Census(BaseModel):
    """
    Census data for user input
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 7777,
                "education": "Some-college",
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Armed-Forces",
                "relationship": "Not-in-family",
                "race": "Other",
                "sex": "Male",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Holand-Netherlands"
            }
        }


app = FastAPI(
    title="Census Inference API",
    description="An API for the Census model",
    version="1.0.0",
)


@app.get("/")
def greet():
    """
    Welcome message
    """
    return {"message": "Welcome!"}


@app.on_event("startup")
async def load_model():
    """
    Event handler that verify that the model files exists
    """
    assert MODEL_FILE.exists()
    assert ENCODER_FILE.exists()
    assert LB_FILE.exists()


@app.post("/predictions/")
async def predict(data: Census):
    """
    Post method to handle live predictions
    """

    input_dataframe = pd.DataFrame(
        {key: val for key, val in data.dict().items()}, index=[0]
    )

    features = ["workclass", "education", "marital_status", "occupation",
                "relationship", "race", "sex", "native_country"]

    model = pickle.load(open(MODEL_FILE, "rb"))
    encoder = pickle.load(open(ENCODER_FILE, "rb"))
    lb = pickle.load(open(LB_FILE, "rb"))

    X, _y, _encoder, _lb = process_data(X=input_dataframe,
                                        categorical_features=features,
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)

    prediction = model.predict(X)

    pred = ">50k" if prediction[0] else "<=50k"

    input_dataframe["prediction"] = "<=50k"
    if prediction[0] > 0.5:
        input_dataframe["prediction"] = ">50k"

    return {"prediction": pred}

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)

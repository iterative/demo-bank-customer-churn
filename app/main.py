from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from jsonschema import validate
from utils.load_params import load_params

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

params = load_params(params_path='params.yaml')
models_dir = Path(params.train.models_dir)
model_fname = params.train.model_fname
model_path = models_dir/model_fname

schema = {
    "type" : "object",
    "properties" : {
        "Geography" : {"type" : "string"},
        "Gender" : {"type" : "string"},
        "CreditScore" : {"type" : "number"},
        "Age" : {"type" : "number"},
        "Tenure" : {"type" : "number"},
        "Balance" : {"type" : "number"},
        "NumOfProducts" : {"type" : "number"},
        "HasCrCard" : {"type" : "number"},
        "IsActiveMember" : {"type" : "number"},
        "EstimatedSalary" : {"type" : "number"}
    },
}

model = load(filename=model_path)

@app.post("/predict")
async def predict(info : Request):
    req_json = await info.json()
    validate(instance=req_json, schema=schema)
    input_data = pd.DataFrame([req_json])
    prob = model.predict_proba(input_data)[0][0]
    return {"prob" : prob}
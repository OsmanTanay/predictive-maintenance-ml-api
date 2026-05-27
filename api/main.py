from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Predictive Maintenance ML API",
    description="API for predicting machine failure based on operating conditions.",
    version="1.0"
)

model = joblib.load("models/random_forest_model.pkl")


class MachineData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int
    type_L: bool
    type_M: bool


@app.post("/predict")
def predict(data: MachineData):
    input_data = [[
        data.air_temperature,
        data.process_temperature,
        data.rotational_speed,
        data.torque,
        data.tool_wear,
        data.type_L,
        data.type_M
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "prediction_label": "Failure" if prediction == 1 else "No Failure"
    }
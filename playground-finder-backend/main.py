from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

le_weekday = encoders["le_weekday"]
le_weather = encoders["le_weather"]
le_crowd = encoders["le_crowd"]

# FastAPI app
app = FastAPI()

# Request model
class PlaygroundRequest(BaseModel):
    playground_id: int
    hour: int
    weekday: str
    weather: str

@app.post("/predict")
def predict_playground_status(data: PlaygroundRequest):
    try:
        weekday_encoded = le_weekday.transform([data.weekday])[0]
        weather_encoded = le_weather.transform([data.weather])[0]

        X = np.array([[data.playground_id, data.hour, weekday_encoded, weather_encoded]])
        pred = model.predict(X)[0]
        crowd_status = le_crowd.inverse_transform([pred])[0]

        return {"status": crowd_status}
    except Exception as e:
        return {"error": str(e)}

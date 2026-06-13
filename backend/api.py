from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor.ann_predictor import predict as predict_ann
from backend.predictor.lstm_predictor import predict as predict_lstm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str
    model: str

def run_prediction(text: str, model: str):
    model = model.strip().lower()

    print(f"Model: {model}, Text: {text}")

    if model == "ann":
        return predict_ann(text)
    return predict_lstm(text)

@app.post("/predict")
def predict_sentiment(request: PredictRequest):
    try:
        result = run_prediction(request.text, request.model)
        return result
    except Exception as e:
        return {
            "error": str(e)
        }


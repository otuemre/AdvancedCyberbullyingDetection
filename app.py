from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AdvancedChatThreatDetection API",
    version="1.0.0",
)

# Load once at startup
pipeline = PredictionPipeline(threshold=0.5, max_len=256)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Message to classify")


class PredictResponse(BaseModel):
    preprocessed_text: str
    prob_cyberbullying: float
    pred_label: int
    pred_name: str


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("POST /predict")
    return pipeline.predict_one(req.text)


@app.post("/predict-batch")
def predict_batch(req: PredictBatchRequest):
    logger.info("POST /predict-batch")
    return {"results": pipeline.predict_batch(req.texts)}

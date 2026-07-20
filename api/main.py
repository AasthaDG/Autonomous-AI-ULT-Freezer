"""
FastAPI service exposing the autonomous AI pipeline as a real-time endpoint.

POST /predict   -> takes a short window of recent sensor readings for one unit,
                    runs perceive -> reason -> act, returns risk score + decision.
GET  /health    -> basic liveness check.

Run:
    uvicorn api.main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
        -d @sample_request.json
"""

from typing import List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.act import decide
from src.perceive import engineer_features
from src.reason import score

app = FastAPI(
    title="Autonomous AI Pipeline - ULT Freezer Predictive Maintenance",
    description="Perceive -> Reason -> Act loop for industrial cold-chain equipment.",
    version="1.0.0",
)


class SensorReading(BaseModel):
    timestamp: str
    unit_id: str
    chamber_temp_c: float
    compressor_current_a: float
    ambient_temp_c: float
    door_open_events: int = 0
    vibration_g: float
    power_draw_w: float


class PredictRequest(BaseModel):
    readings: List[SensorReading] = Field(
        ..., description="Recent readings for one unit, oldest first, at least 24 hourly points recommended."
    )


class PredictResponse(BaseModel):
    unit_id: str
    risk_score: float
    action: str
    reason: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.readings])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["failed"] = 0
    df["at_risk"] = 0

    features = engineer_features(df)
    latest = features.iloc[[-1]].copy()
    latest["risk_score"] = score(latest)

    d = decide(
        risk_score=float(latest["risk_score"].iloc[0]),
        temp_drift=float(latest["chamber_temp_c_drift"].iloc[0]),
        current_drift=float(latest["compressor_current_a_drift"].iloc[0]),
    )

    return PredictResponse(
        unit_id=latest["unit_id"].iloc[0],
        risk_score=round(d.risk_score, 4),
        action=d.action,
        reason=d.reason,
    )

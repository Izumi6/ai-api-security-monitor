from fastapi import FastAPI, Request
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

MODEL_PATH = Path("models/request_iforest.joblib")
SCALER_PATH = Path("models/request_scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(
    title="AI API Security Monitor",
    description="Monitors ML/AI API traffic and flags anomalous requests.",
    version="1.0.0",
)


class LoggedRequest(BaseModel):
    timestamp: str
    ip: str
    endpoint: str
    method: str
    bytes_in: int
    bytes_out: int
    status_code: int
    latency_ms: int


def build_feature_vector(r: LoggedRequest) -> np.ndarray:
    ip_octet_1 = int(r.ip.split(".")[0]) if "." in r.ip else 0
    method_code = 1 if r.method.upper() == "POST" else 0
    endpoint_code = 0 if r.endpoint == "/predict" else 1

    x = np.array(
        [
            [
                endpoint_code,
                method_code,
                r.bytes_in,
                r.bytes_out,
                r.status_code,
                r.latency_ms,
                ip_octet_1,
            ]
        ]
    )
    return x


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI API Security Monitor running"}


@app.post("/monitor")
async def monitor_request(r: LoggedRequest):
    x = build_feature_vector(r)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    score = model.decision_function(x_scaled)[0]
    is_anomaly = pred == -1

    label = "Normal"
    if is_anomaly:
        label = "Suspicious"

    risk_level = "Low"
    if is_anomaly and score < -0.2:
        risk_level = "High"
    elif is_anomaly:
        risk_level = "Medium"

    summary = {
        "label": label,
        "risk_level": risk_level,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly),
    }

    explanation = []
    if is_anomaly:
        explanation.append("Request pattern deviates from typical traffic.")
        if r.bytes_in > 4000:
            explanation.append("Unusually large request body (possible scraping or abuse).")
        if r.latency_ms > 800:
            explanation.append("High latency, may indicate heavy or abusive payload.")
    else:
        explanation.append("Request appears similar to normal traffic.")

    return {
        "summary": summary,
        "explanation": explanation,
        "raw_request": r.dict(),
    }

"""
VoltShift-RL: Production Edge Dispatch Microservice.
Exposes a low-latency REST API via FastAPI for physical SCADA integration.
Runs real-time ML forecasting, PPO policy inference, and safety shield clamping.
"""

from contextlib import asynccontextmanager
from typing import List, Dict, Any
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

from src.config import (
    RL_MODEL_PATH,
    FORECASTER_MODEL_PATH,
    TRANSFORMER_SANCTIONED_LIMIT_KW,
    BESS_CAPACITY_KWH,
    BESS_MIN_RESERVE_KWH,
    TARIFF_OFF_PEAK,
    TARIFF_NORMAL,
    TARIFF_PEAK
)
from src.safety_shield import SafetyShield

# -------------------------------------------------------------
# 1. TARIFF UTILITY FUNCTION
# -------------------------------------------------------------
def get_current_tariff(hour: float) -> float:
    """Returns the DISCOM ToD tariff slab based on time of day."""
    if 22.0 <= hour or hour < 6.0:
        return TARIFF_OFF_PEAK
    elif 18.0 <= hour < 22.0:
        return TARIFF_PEAK
    else:
        return TARIFF_NORMAL

# -------------------------------------------------------------
# 2. PYDANTIC REQUEST & RESPONSE SCHEMAS
# -------------------------------------------------------------
class MicrogridTelemetryRequest(BaseModel):
    hour: float = Field(..., ge=0.0, lt=24.0, description="Local clock hour (e.g., 18.25 for 18:15)")
    ambient_temp_c: float = Field(..., ge=10.0, le=55.0, description="Depot temperature in Celsius")
    solar_current_kw: float = Field(..., ge=0.0, le=25.0, description="Current solar output (kW)")
    solar_lag_15m_kw: float = Field(..., ge=0.0, le=25.0, description="Solar 15m ago (kW)")
    solar_lag_30m_kw: float = Field(..., ge=0.0, le=25.0, description="Solar 30m ago (kW)")
    solar_lag_60m_kw: float = Field(..., ge=0.0, le=25.0, description="Solar 60m ago (kW)")
    bess_soc_kwh: float = Field(..., ge=BESS_MIN_RESERVE_KWH, le=BESS_CAPACITY_KWH, description="BESS SoC (kWh)")
    ev_soc_kwh: List[float] = Field(..., min_length=3, max_length=3, description="EV SoC for bays 1, 2, 3 (kWh)")

class DispatchResponse(BaseModel):
    status: str
    active_tariff_inr: float
    predicted_solar_1h_kw: float
    clamped_bess_power_kw: float
    clamped_ev_power_kw: List[float]
    projected_grid_draw_kw: float
    grid_transformer_headroom_kw: float
    shield_interventions: Dict[str, bool]
    execution_time_ms: float

# -------------------------------------------------------------
# 3. MODEL LIFECYCLE & CACHING
# -------------------------------------------------------------
models: Dict[str, Any] = {}

def load_all_models():
    """Idempotently loads and caches model artifacts in memory."""
    if "forecaster" not in models:
        models["forecaster"] = joblib.load(FORECASTER_MODEL_PATH)
        models["ppo"] = PPO.load(str(RL_MODEL_PATH), device="cpu")
        models["shield"] = SafetyShield()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- Loading VoltShift Production Inference Engine ---")
    load_all_models()
    print("Supervised Forecaster, Continuous PPO Policy, and Safety Shield loaded.\n")
    yield
    models.clear()

app = FastAPI(
    title="VoltShift-RL Microgrid Dispatch Gateway",
    description="Low-latency autonomous dispatch engine for solar-powered EV fleet depots with BESS.",
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------------------------------------
# 4. ENDPOINTS
# -------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Redirect base URL to Swagger docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    load_all_models()
    return {
        "status": "ONLINE",
        "models_loaded": list(models.keys()),
        "sanctioned_transformer_limit_kw": TRANSFORMER_SANCTIONED_LIMIT_KW
    }

@app.post("/dispatch", response_model=DispatchResponse)
async def compute_dispatch_setpoints(payload: MicrogridTelemetryRequest):
    load_all_models()
    start_time = time.perf_counter()

    try:
        tariff = get_current_tariff(payload.hour)

        # 1-hour ahead solar forecast inference
        forecast_df = pd.DataFrame([{
            "hour": payload.hour,
            "ambient_temp_c": payload.ambient_temp_c,
            "solar_current": payload.solar_current_kw,
            "solar_lag_15m": payload.solar_lag_15m_kw,
            "solar_lag_30m": payload.solar_lag_30m_kw,
            "solar_lag_60m": payload.solar_lag_60m_kw
        }])
        predicted_solar_1h = max(0.0, float(models["forecaster"].predict(forecast_df)[0]))

        # 8-dimensional observation vector for PPO
        obs = np.array([
            payload.hour,
            payload.solar_current_kw,
            tariff,
            predicted_solar_1h,
            payload.bess_soc_kwh,
            payload.ev_soc_kwh[0],
            payload.ev_soc_kwh[1],
            payload.ev_soc_kwh[2]
        ], dtype=np.float32)

        # PPO deterministic action inference
        raw_action, _ = models["ppo"].predict(obs, deterministic=True)
        raw_bess = float(raw_action[0])
        raw_evs = np.array(raw_action[1:], dtype=np.float32)

        # Fleet presence constraint: charging active only when fleet is docked (18:00 to 06:00)
        fleet_at_depot = (payload.hour >= 18.0 or payload.hour < 6.0)
        if not fleet_at_depot:
            raw_evs = np.zeros(3, dtype=np.float32)

        # Enforce non-bypassable safety envelope
        clamped_bess_kw, clamped_ev_kw, projected_grid_kw, interventions = models["shield"].filter_actions(
            raw_bess_action=raw_bess,
            raw_ev_actions=raw_evs,
            solar_kw=payload.solar_current_kw,
            current_bess_kwh=payload.bess_soc_kwh,
            current_ev_kwh=np.array(payload.ev_soc_kwh, dtype=np.float32)
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DispatchResponse(
            status="SUCCESS",
            active_tariff_inr=tariff,
            predicted_solar_1h_kw=round(predicted_solar_1h, 2),
            clamped_bess_power_kw=round(float(clamped_bess_kw), 2),
            clamped_ev_power_kw=[round(float(p), 2) for p in clamped_ev_kw],
            projected_grid_draw_kw=round(float(projected_grid_kw), 2),
            grid_transformer_headroom_kw=round(TRANSFORMER_SANCTIONED_LIMIT_KW - projected_grid_kw, 2),
            shield_interventions=interventions,
            execution_time_ms=round(elapsed_ms, 2)
        )

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
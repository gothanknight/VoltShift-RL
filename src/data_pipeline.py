"""
Data ingestion and telemetry generation pipeline for VoltShift-RL.
Generates 30 days of 15-minute empirical solar, temperature, and tariff logs
with correlated cloud attenuation, saving directly to Apache Parquet.
"""

import numpy as np
import polars as pl
from src.config import (
    TELEMETRY_PARQUET_PATH,
    TOTAL_SIMULATION_DAYS,
    STEPS_PER_DAY,
    DT_HOURS,
    TARIFF_OFF_PEAK,
    TARIFF_NORMAL,
    TARIFF_PEAK,
    PEAK_WINDOW_START,
    PEAK_WINDOW_END,
    OFF_PEAK_WINDOW_START,
    OFF_PEAK_WINDOW_END,
)

def compute_solar_irradiance(hour: float, cloud_factor: float) -> float:
    """
    Computes solar irradiance using a clear-sky insolation curve
    modulated by stochastic cloud extinction.
    """
    if 6.0 <= hour <= 18.0:
        # Solar elevation curve peaking at 12:00 PM (~14.5 kW clear sky ceiling)
        clear_sky_kw = 14.5 * np.sin((hour - 6.0) / 12.0 * np.pi)
        return float(max(0.0, clear_sky_kw * cloud_factor))
    return 0.0

def compute_tariff(hour: float) -> float:
    """Returns the commercial ToD tariff for a given hour."""
    if PEAK_WINDOW_START <= hour < PEAK_WINDOW_END:
        return TARIFF_PEAK
    elif hour >= OFF_PEAK_WINDOW_START or hour < OFF_PEAK_WINDOW_END:
        return TARIFF_OFF_PEAK
    return TARIFF_NORMAL

def generate_telemetry_dataset() -> pl.DataFrame:
    """
    Synthesizes 30 days of 15-minute resolution depot telemetry.
    Uses an autoregressive Markov-like cloud model to simulate realistic
    weather fronts rather than independent random noise.
    """
    np.random.seed(42)  # Fixed seed for reproducibility
    total_steps = TOTAL_SIMULATION_DAYS * STEPS_PER_DAY

    records = []
    current_cloud_factor = 1.0  # 1.0 = clear sky, 0.2 = heavy cloud cover

    for step in range(total_steps):
        day = step // STEPS_PER_DAY
        hour = (step % STEPS_PER_DAY) * DT_HOURS

        # Correlated cloud cover (weather persists across consecutive steps)
        if np.random.rand() < 0.08:
            # Weather transition event (storm/cloud front enters or clears)
            current_cloud_factor = np.random.uniform(0.25, 0.65) if np.random.rand() < 0.20 else np.random.uniform(0.85, 1.0)
        else:
            # Slight random walk around current weather state
            current_cloud_factor = np.clip(current_cloud_factor + np.random.normal(0, 0.02), 0.20, 1.0)

        solar_kw = compute_solar_irradiance(hour, current_cloud_factor)

        # Diurnal thermal lag: temperature peaks ~2.5 hours after solar noon
        ambient_temp_c = 23.0 + 15.0 * np.sin((hour - 8.5) / 12.0 * np.pi) + np.random.normal(0, 0.4)

        tariff = compute_tariff(hour)

        records.append({
            "step": step,
            "day": int(day),
            "hour": float(round(hour, 2)),
            "solar_kw": float(round(solar_kw, 3)),
            "ambient_temp_c": float(round(ambient_temp_c, 2)),
            "cloud_factor": float(round(current_cloud_factor, 3)),
            "tariff": float(tariff),
        })

    df = pl.DataFrame(records)
    
    # Save directly to processed parquet directory
    df.write_parquet(TELEMETRY_PARQUET_PATH)
    return df

if __name__ == "__main__":
    df = generate_telemetry_dataset()
    print("\n--- Telemetry Generation Summary ---")
    print(f"File Saved To: {TELEMETRY_PARQUET_PATH}")
    print(f"Total Rows:     {len(df)} (30 days @ 96 steps/day)")
    print(f"Schema:\n{df.schema}\n")
    print("Sample Telemetry (Noon Peak vs Evening Peak):")
    # Display rows at noon (12:00) and evening peak (19:00)
    sample = df.filter(pl.col("hour").is_in([12.0, 19.0])).head(4)
    print(sample)
"""
Central configuration for VoltShift-RL.
Defines all directory paths, physical hardware constraints,
tariff schedules, and simulation hyperparameters.
"""

from pathlib import Path

# ============================================================================
# 1. DIRECTORY PATHS
# ============================================================================
# Project root directory (resolves dynamically from this file's location)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Standard file targets
TELEMETRY_PARQUET_PATH = DATA_PROCESSED_DIR / "depot_telemetry.parquet"
FORECASTER_MODEL_PATH = MODELS_DIR / "solar_forecaster.joblib"
RL_MODEL_PATH = MODELS_DIR / "ppo_voltshift_agent"

# ============================================================================
# 2. TIME & SIMULATION CONSTANTS
# ============================================================================
TIMESTEP_MINUTES = 15                    # Sensor resolution
DT_HOURS = TIMESTEP_MINUTES / 60.0       # 0.25 hours per step
STEPS_PER_DAY = int(24 / DT_HOURS)       # 96 steps in a full 24-hour cycle
TOTAL_SIMULATION_DAYS = 30               # Length of historical dataset

# ============================================================================
# 3. STATIONARY BESS (BATTERY ENERGY STORAGE SYSTEM)
# ============================================================================
BESS_CAPACITY_KWH = 30.0                 # Total physical storage capacity
BESS_MIN_RESERVE_KWH = 3.0               # 10% depth-of-discharge reserve
BESS_MAX_POWER_KW = 8.0                  # Max charge / discharge throttle (inverter cap)
BESS_ROUNDTRIP_EFFICIENCY = 0.92         # 92% efficiency (thermal/inverter losses)

# ============================================================================
# 4. COMMERCIAL EV FLEET SPECIFICATIONS
# ============================================================================
NUM_EVS = 3                              # 3 commercial delivery vehicles (e.g., 3-wheelers)
EV_BATTERY_CAPACITY_KWH = 10.0           # 10 kWh battery pack per vehicle
EV_TARGET_READY_KWH = 9.5                # 95% charge required for morning dispatch
EV_MAX_CHARGING_POWER_KW = 5.0           # Level 2 commercial AC charger ceiling
EV_CHARGING_EFFICIENCY = 0.92            # Onboard charger AC-to-DC efficiency

# Fleet schedule
EV_ARRIVAL_HOUR = 18.0                   # Returns to hub low on charge at 6:00 PM
EV_DEPARTURE_HOUR = 6.0                  # Must be fully charged by 6:00 AM

# ============================================================================
# 5. GRID & TIME-OF-DAY (ToD) TARIFFS (GUJARAT/DISCOM CALIBRATION)
# ============================================================================
TRANSFORMER_SANCTIONED_LIMIT_KW = 15.0   # Peak allowable grid draw before tripping

# Tariff rates (₹ per kWh)
TARIFF_OFF_PEAK = 4.20                   # Night off-peak rate (22:00 to 06:00)
TARIFF_NORMAL = 7.50                     # Daytime operational rate (06:00 to 18:00)
TARIFF_PEAK = 12.50                      # Evening peak demand rate (18:00 to 22:00)

PEAK_WINDOW_START = 18.0
PEAK_WINDOW_END = 22.0
OFF_PEAK_WINDOW_START = 22.0
OFF_PEAK_WINDOW_END = 6.0

# ============================================================================
# 6. REWARD WEIGHTS & PENALTIES
# ============================================================================
PENALTY_TRANSFORMER_BREACH = 25.0        # Multiplier on excess kW above transformer limit
PENALTY_BATTERY_WEAR = 0.05              # Micro-cost on BESS usage to prevent jitter
PENALTY_FLEET_UNREADY = 50.0             # Cost per missing kWh if EV is not ready at 6 AM
REWARD_GREEN_CAPTURE = 1.5               # Incentive for buffering surplus solar
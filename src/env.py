"""
Data-driven Gymnasium Environment for VoltShift-RL.
Supports multi-day horizons (1 to 30 days) with realistic daily fleet cycles.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import pandas as pd
import joblib

from src.config import (
    TELEMETRY_PARQUET_PATH,
    FORECASTER_MODEL_PATH,
    BESS_CAPACITY_KWH,
    BESS_MIN_RESERVE_KWH,
    BESS_ROUNDTRIP_EFFICIENCY,
    EV_BATTERY_CAPACITY_KWH,
    EV_TARGET_READY_KWH,
    EV_CHARGING_EFFICIENCY,
    STEPS_PER_DAY,
    DT_HOURS,
    PENALTY_BATTERY_WEAR,
    PENALTY_FLEET_UNREADY,
    REWARD_GREEN_CAPTURE,
)
from src.safety_shield import SafetyShield

class VoltShiftDepotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        telemetry_path=TELEMETRY_PARQUET_PATH,
        forecaster_path=FORECASTER_MODEL_PATH,
        episode_days: int = 1
    ):
        super().__init__()

        self.telemetry = pl.read_parquet(telemetry_path).to_pandas()
        self.forecaster = joblib.load(forecaster_path)
        self.shield = SafetyShield()
        
        self.total_rows = len(self.telemetry)
        self.episode_days = episode_days
        self.max_episode_steps = self.episode_days * STEPS_PER_DAY

        # Action Space: [-1, 1] for BESS, [0, 1] for 3 EV bays
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space (8 continuous values)
        obs_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([24.0, 20.0, 20.0, 20.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.reset()

    def _get_ml_forecast(self, row_idx: int) -> float:
        row = self.telemetry.iloc[row_idx]
        lag1 = self.telemetry.iloc[max(0, row_idx - 1)]["solar_kw"]
        lag2 = self.telemetry.iloc[max(0, row_idx - 2)]["solar_kw"]
        lag4 = self.telemetry.iloc[max(0, row_idx - 4)]["solar_kw"]

        features = pd.DataFrame([{
            "hour": row["hour"],
            "ambient_temp_c": row["ambient_temp_c"],
            "solar_current": row["solar_kw"],
            "solar_lag_15m": lag1,
            "solar_lag_30m": lag2,
            "solar_lag_60m": lag4
        }])

        forecast = float(self.forecaster.predict(features)[0])
        return max(0.0, forecast)

    def _get_obs(self) -> np.ndarray:
        row = self.telemetry.iloc[self.current_idx]
        forecast_solar = self._get_ml_forecast(self.current_idx)

        return np.array([
            row["hour"],
            row["solar_kw"],
            row["tariff"],
            forecast_solar,
            self.bess_kwh,
            self.ev_soc[0],
            self.ev_soc[1],
            self.ev_soc[2]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start_idx = self.total_rows - self.max_episode_steps
        if seed is not None:
            np.random.seed(seed)
            self.current_idx = (seed * STEPS_PER_DAY) % max_start_idx
        else:
            max_start_day = max(1, (max_start_idx // STEPS_PER_DAY))
            random_day = np.random.randint(0, max_start_day)
            self.current_idx = random_day * STEPS_PER_DAY

        self.step_in_episode = 0
        self.bess_kwh = float(np.random.uniform(5.0, 10.0))
        self.ev_soc = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        self.cumulative_cost = 0.0
        self.cumulative_solar_used = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        row = self.telemetry.iloc[self.current_idx]
        hour = float(row["hour"])
        solar_kw = float(row["solar_kw"])
        tariff = float(row["tariff"])

        # Check if fleet is at depot (18:00 to 06:00)
        fleet_at_depot = (hour >= 18.0 or hour < 6.0)

        # -------------------------------------------------------------
        # 1. Daily Fleet Logistics Cycle Trigger (Arrival at 18:00)
        # -------------------------------------------------------------
        # At exactly 18:00 (step % 96 == 72), vehicles return depleted from delivery routes
        if self.step_in_episode > 0 and (self.step_in_episode % STEPS_PER_DAY == 72):
            self.ev_soc = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        raw_bess_action = float(action[0])
        # If fleet is away on routes, EV charging throttle is forced to 0
        raw_ev_actions = np.array(action[1:], dtype=np.float32) if fleet_at_depot else np.zeros(3, dtype=np.float32)

        # -------------------------------------------------------------
        # 2. Deterministic Safety Shield
        # -------------------------------------------------------------
        clamped_bess_kw, clamped_ev_kw, grid_draw_kw, interventions = self.shield.filter_actions(
            raw_bess_action=raw_bess_action,
            raw_ev_actions=raw_ev_actions,
            solar_kw=solar_kw,
            current_bess_kwh=self.bess_kwh,
            current_ev_kwh=self.ev_soc
        )

        # -------------------------------------------------------------
        # 3. Update Storage Assets
        # -------------------------------------------------------------
        if clamped_bess_kw > 0.0:
            self.bess_kwh += clamped_bess_kw * DT_HOURS * BESS_ROUNDTRIP_EFFICIENCY
        elif clamped_bess_kw < 0.0:
            self.bess_kwh -= (abs(clamped_bess_kw) * DT_HOURS) / BESS_ROUNDTRIP_EFFICIENCY
        self.bess_kwh = float(np.clip(self.bess_kwh, BESS_MIN_RESERVE_KWH, BESS_CAPACITY_KWH))

        if fleet_at_depot:
            for i in range(len(self.ev_soc)):
                charge_added = clamped_ev_kw[i] * DT_HOURS * EV_CHARGING_EFFICIENCY
                self.ev_soc[i] = min(EV_BATTERY_CAPACITY_KWH, self.ev_soc[i] + charge_added)

        step_cost = grid_draw_kw * DT_HOURS * tariff
        self.cumulative_cost += step_cost

        # -------------------------------------------------------------
        # 4. Multi-Objective Reward
        # -------------------------------------------------------------
        reward = -step_cost

        if clamped_bess_kw > 0.0 and solar_kw > 0.0:
            solar_captured = min(clamped_bess_kw, solar_kw) * DT_HOURS
            reward += solar_captured * REWARD_GREEN_CAPTURE
            self.cumulative_solar_used += solar_captured

        reward -= abs(clamped_bess_kw) * PENALTY_BATTERY_WEAR

        # Check morning dispatch readiness penalty at 06:00 (step % 96 == 24)
        if self.step_in_episode > 0 and (self.step_in_episode % STEPS_PER_DAY == 24):
            deficit = float(np.sum([max(0.0, EV_TARGET_READY_KWH - soc) for soc in self.ev_soc]))
            reward -= deficit * PENALTY_FLEET_UNREADY

        # -------------------------------------------------------------
        # 5. Advance Timestep
        # -------------------------------------------------------------
        self.current_idx += 1
        self.step_in_episode += 1

        terminated = bool(self.step_in_episode >= self.max_episode_steps)

        info = {
            "hour": hour,
            "step_in_episode": self.step_in_episode,
            "solar_kw": solar_kw,
            "tariff": tariff,
            "grid_draw_kw": grid_draw_kw,
            "bess_kw": clamped_bess_kw,
            "ev_kw": clamped_ev_kw,
            "step_cost": step_cost,
            "cumulative_cost": self.cumulative_cost,
            "bess_soc": self.bess_kwh,
            "fleet_soc": np.copy(self.ev_soc),
            "fleet_at_depot": fleet_at_depot,
            "interventions": interventions
        }

        return self._get_obs(), float(reward), terminated, False, info
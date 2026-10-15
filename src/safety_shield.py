"""
Deterministic Rule-Based Safety Shield for VoltShift-RL.
Acts as a non-bypassable electrical governor enforcing physical battery bounds
and the DISCOM sanctioned transformer capacity ceiling.
"""

import numpy as np
from src.config import (
    TRANSFORMER_SANCTIONED_LIMIT_KW,
    BESS_CAPACITY_KWH,
    BESS_MIN_RESERVE_KWH,
    BESS_MAX_POWER_KW,
    BESS_ROUNDTRIP_EFFICIENCY,
    EV_BATTERY_CAPACITY_KWH,
    EV_MAX_CHARGING_POWER_KW,
    EV_CHARGING_EFFICIENCY,
    DT_HOURS
)

class SafetyShield:
    def __init__(
        self,
        transformer_limit_kw: float = TRANSFORMER_SANCTIONED_LIMIT_KW,
        bess_capacity_kwh: float = BESS_CAPACITY_KWH,
        bess_min_reserve_kwh: float = BESS_MIN_RESERVE_KWH,
        bess_max_power_kw: float = BESS_MAX_POWER_KW
    ):
        self.transformer_limit_kw = transformer_limit_kw
        self.bess_capacity_kwh = bess_capacity_kwh
        self.bess_min_reserve_kwh = bess_min_reserve_kwh
        self.bess_max_power_kw = bess_max_power_kw

    def filter_actions(
        self,
        raw_bess_action: float,
        raw_ev_actions: np.ndarray,
        solar_kw: float,
        current_bess_kwh: float,
        current_ev_kwh: np.ndarray
    ):
        """
        Clamps continuous agent action setpoints to physical and contractual limits.
        
        Parameters:
            raw_bess_action: Float in [-1.0, 1.0] representing BESS power demand.
            raw_ev_actions: Array of shape (3,) in [0.0, 1.0] for EV bays.
            solar_kw: Realized solar power at current timestep.
            current_bess_kwh: Active BESS charge level (kWh).
            current_ev_kwh: Active fleet SoC array of shape (3,) (kWh).
            
        Returns:
            clamped_bess_kw: Executable BESS power setpoint (kW, + is charge, - is discharge).
            clamped_ev_kw: Executable EV charging array of shape (3,) (kW).
            projected_grid_kw: Resulting grid draw (kW) guaranteed <= transformer limit.
            interventions: Dict logging whether the shield was triggered.
        """
        interventions = {
            "bess_overcharge_clamped": False,
            "bess_deep_discharge_clamped": False,
            "transformer_limit_clamped": False
        }

        # -------------------------------------------------------------
        # 1. Denormalize Raw Actions to Desired Kilowatts
        # -------------------------------------------------------------
        desired_bess_kw = float(np.clip(raw_bess_action, -1.0, 1.0) * self.bess_max_power_kw)
        desired_ev_kw = np.clip(raw_ev_actions, 0.0, 1.0) * EV_MAX_CHARGING_POWER_KW

        # -------------------------------------------------------------
        # 2. Enforce EV Battery Absorption Limits
        # -------------------------------------------------------------
        # Vehicles cannot absorb more energy than their empty headroom
        clamped_ev_kw = np.zeros_like(desired_ev_kw)
        for i in range(len(desired_ev_kw)):
            headroom_kwh = max(0.0, EV_BATTERY_CAPACITY_KWH - current_ev_kwh[i])
            max_absorbable_kw = headroom_kwh / (DT_HOURS * EV_CHARGING_EFFICIENCY)
            clamped_ev_kw[i] = min(desired_ev_kw[i], max_absorbable_kw)

        # -------------------------------------------------------------
        # 3. Enforce BESS Chemical Charge / Discharge Invariants
        # -------------------------------------------------------------
        clamped_bess_kw = desired_bess_kw
        
        if desired_bess_kw > 0.0:
            # Charging: cannot exceed remaining battery capacity
            headroom_kwh = max(0.0, self.bess_capacity_kwh - current_bess_kwh)
            max_charge_kw = headroom_kwh / (DT_HOURS * BESS_ROUNDTRIP_EFFICIENCY)
            if desired_bess_kw > max_charge_kw:
                clamped_bess_kw = max_charge_kw
                interventions["bess_overcharge_clamped"] = True
                
        elif desired_bess_kw < 0.0:
            # Discharging: cannot drop below minimum 10% reserve threshold
            available_kwh = max(0.0, current_bess_kwh - self.bess_min_reserve_kwh)
            max_discharge_kw = (available_kwh * BESS_ROUNDTRIP_EFFICIENCY) / DT_HOURS
            if abs(desired_bess_kw) > max_discharge_kw:
                clamped_bess_kw = -max_discharge_kw
                interventions["bess_deep_discharge_clamped"] = True

        # -------------------------------------------------------------
        # 4. Enforce Transformer Sanctioned Limit Clamping
        # -------------------------------------------------------------
        # Local supply generated inside the depot (Solar + BESS discharge)
        bess_discharge_kw = max(0.0, -clamped_bess_kw)
        bess_charge_kw = max(0.0, clamped_bess_kw)
        
        local_generation_kw = solar_kw + bess_discharge_kw
        total_demand_kw = np.sum(clamped_ev_kw) + bess_charge_kw

        # Grid draw is the deficit after using on-site generation
        projected_grid_kw = max(0.0, total_demand_kw - local_generation_kw)

        # If projected draw trips the transformer, scale back demand
        if projected_grid_kw > self.transformer_limit_kw:
            interventions["transformer_limit_clamped"] = True
            excess_kw = projected_grid_kw - self.transformer_limit_kw
            
            # Priority 1: Cancel any grid charging of stationary BESS
            if bess_charge_kw > 0.0:
                reduction = min(bess_charge_kw, excess_kw)
                clamped_bess_kw -= reduction
                excess_kw -= reduction

            # Priority 2: Proportionally throttle EV charging bays if still over limit
            if excess_kw > 0.0 and np.sum(clamped_ev_kw) > 0.0:
                scale_ratio = max(0.0, (np.sum(clamped_ev_kw) - excess_kw) / np.sum(clamped_ev_kw))
                clamped_ev_kw = clamped_ev_kw * scale_ratio

            # Recompute guaranteed safe grid load
            final_demand_kw = np.sum(clamped_ev_kw) + max(0.0, clamped_bess_kw)
            projected_grid_kw = max(0.0, final_demand_kw - local_generation_kw)

        return clamped_bess_kw, clamped_ev_kw, float(projected_grid_kw), interventions

if __name__ == "__main__":
    # Self-test unit verification
    shield = SafetyShield()
    
    # Stress Case 1: Agent tries to draw 25 kW on a 15 kW grid line with 0 solar
    raw_bess = 1.0  # Wants +8 kW charge
    raw_evs = np.array([1.0, 1.0, 1.0])  # Wants 5 kW each = 15 kW
    solar = 0.0
    bess_soc = 10.0
    ev_soc = np.array([2.0, 2.0, 2.0])

    safe_bess, safe_evs, grid_draw, flags = shield.filter_actions(
        raw_bess, raw_evs, solar, bess_soc, ev_soc
    )

    print("\n--- Safety Shield Stress Test ---")
    print(f"Raw Demanded Power:   {8.0 + 15.0:.2f} kW")
    print(f"Clamped Grid Draw:    {grid_draw:.2f} kW (Must be <= {shield.transformer_limit_kw} kW)")
    print(f"Clamped BESS Setpoint:{safe_bess:.2f} kW")
    print(f"Clamped EV Setpoints: {safe_evs} kW")
    print(f"Shield Interventions: {flags}")
    assert grid_draw <= shield.transformer_limit_kw + 1e-5, "Transformer breach detected!"
    print("Safety Shield Invariant Check: PASSED\n")
"""
PPO Policy Training & Benchmark Evaluation for VoltShift-RL.
Trains a continuous Actor-Critic agent on CPU and evaluates
performance against an unmanaged baseline.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import (
    RL_MODEL_PATH,
    STEPS_PER_DAY,
    DT_HOURS,
    TARIFF_PEAK,
    TRANSFORMER_SANCTIONED_LIMIT_KW
)
from src.env import VoltShiftDepotEnv

# Force lightweight CPU execution with 4 threads
torch.set_num_threads(4)

def evaluate_baseline_vs_rl(env: VoltShiftDepotEnv, model: PPO, test_seed: int = 25):
    """
    Simulates a held-out test day comparing:
    1. Unmanaged Greedy Baseline (charges immediately at full 5 kW until 100% full)
    2. Trained Continuous PPO Agent (VoltShift-RL)
    """
    print("\n" + "="*65)
    print(f"--- RUNNING BENCHMARK EVALUATION (Test Day Seed: {test_seed}) ---")
    print("="*65)

    # ---------------------------------------------------------
    # 1. Simulate Unmanaged Greedy Baseline (State tracked locally)
    # ---------------------------------------------------------
    obs, _ = env.reset(seed=test_seed)
    baseline_cost = 0.0
    baseline_peak_grid_draw = 0.0
    baseline_transformer_breaches = 0
    simulated_ev_soc = np.array([2.0, 2.0, 2.0], dtype=np.float32)

    for _ in range(STEPS_PER_DAY):
        hour = obs[0]
        tariff = obs[2]
        solar = obs[1]
        
        # Greedy logic: active only when fleet returns to depot (18:00 onwards)
        grid_kw = 0.0
        if hour >= 18.0 or hour < 6.0:
            for i in range(3):
                headroom = max(0.0, 10.0 - simulated_ev_soc[i])
                if headroom > 0.05:
                    charge_rate = min(5.0, headroom / (DT_HOURS * 0.92))
                    simulated_ev_soc[i] += charge_rate * DT_HOURS * 0.92
                    grid_kw += charge_rate

        baseline_peak_grid_draw = max(baseline_peak_grid_draw, grid_kw)
        if grid_kw > TRANSFORMER_SANCTIONED_LIMIT_KW:
            baseline_transformer_breaches += 1

        baseline_cost += grid_kw * DT_HOURS * tariff
        
        obs, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        if terminated:
            break

    # ---------------------------------------------------------
    # 2. Simulate Trained PPO Agent
    # ---------------------------------------------------------
    obs, _ = env.reset(seed=test_seed)
    rl_cost = 0.0
    rl_peak_grid_draw = 0.0
    rl_transformer_breaches = 0
    rl_bess_discharged = 0.0

    for _ in range(STEPS_PER_DAY):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        
        grid_draw = info["grid_draw_kw"]
        rl_peak_grid_draw = max(rl_peak_grid_draw, grid_draw)
        
        if grid_draw > TRANSFORMER_SANCTIONED_LIMIT_KW:
            rl_transformer_breaches += 1
            
        if info["bess_kw"] < 0.0:
            rl_bess_discharged += abs(info["bess_kw"]) * DT_HOURS

        if terminated:
            rl_cost = info["cumulative_cost"]
            break

    # ---------------------------------------------------------
    # 3. Print Realistic Quantitative Comparison
    # ---------------------------------------------------------
    savings_inr = max(0.0, baseline_cost - rl_cost)
    savings_pct = (savings_inr / max(1e-5, baseline_cost)) * 100.0

    print(f"{'Performance Metric':<32} | {'Unmanaged Baseline':<16} | {'VoltShift PPO':<14}")
    print("-" * 68)
    print(f"{'Daily Grid Electricity Bill':<32} | ₹{baseline_cost:<15.2f} | ₹{rl_cost:<13.2f}")
    print(f"{'Peak Grid Draw (kW)':<32} | {baseline_peak_grid_draw:<16.2f} | {rl_peak_grid_draw:<14.2f}")
    print(f"{'Transformer Breaches (>15 kW)':<32} | {baseline_transformer_breaches:<16} | {rl_transformer_breaches:<14}")
    print(f"{'BESS Energy Dispatched':<32} | {'0.00 kWh':<16} | {rl_bess_discharged:<11.2f} kWh")
    print("-" * 68)
    print(f"NET RUPEE SAVINGS: ₹{savings_inr:.2f} ({savings_pct:.1f}% Reduction)")
    print("="*65 + "\n")


def train():
    print("\n--- Initializing VoltShift Continuous PPO Training Pipeline ---")
    
    # Vectorized single-environment wrapper
    train_env = DummyVecEnv([lambda: VoltShiftDepotEnv()])

    # Lightweight PPO Policy configured for CPU
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu"
    )

    print("Beginning 80,000 timesteps of training on CPU...")
    model.learn(total_timesteps=80000)

    # Save model artifact
    model.save(str(RL_MODEL_PATH))
    print(f"Model checkpoint saved successfully to: {RL_MODEL_PATH}.zip")

    # Run side-by-side benchmark test
    eval_env = VoltShiftDepotEnv()
    evaluate_baseline_vs_rl(eval_env, model, test_seed=25)

if __name__ == "__main__":
    train()
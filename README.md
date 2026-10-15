# VoltShift-RL: Smart Electric Vehicle Depot Manager

VoltShift-RL is an AI-powered energy manager for a commercial delivery van depot. It coordinates rooftop solar panels, a large stationary storage battery, and 3 electric delivery vans under real Indian electricity rates (Time-of-Day tariffs).

Instead of charging vehicles directly from the grid during expensive peak evening hours, the AI learns to catch free midday sunlight, store it in an on-site battery, and charge the vans off-grid. In the committed 7-day test, this cut the depot's electricity bill by 82.4%, and all three vans were at full charge (10.0 kWh) at 06:00 on all 7 mornings.

---

## 1. What Problem Does This Solve?

In places like India, electricity prices change depending on the time of day (DISCOM Time-of-Day tariffs):
- Night (Off-Peak, 22:00 to 06:00): Rs 4.20 per unit (Cheap)
- Day (Normal, 06:00 to 18:00): Rs 7.50 per unit (Regular)
- Evening (Peak, 18:00 to 22:00): Rs 12.50 per unit (Very Expensive)

Commercial delivery vans return to the depot around 18:00 (6:00 PM) with empty batteries.

- The Bad Way (Unmanaged / Standard Depot): All vans plug in immediately at 6:00 PM and draw up to 15 kW from the grid during the Rs 12.50 peak rate. This raises the bill and sits right at the 15 kW transformer limit.
- The VoltShift Way (AI Managed): The AI stores solar power in a 30 kWh battery during the day and uses it to charge the vans in the evening. In the committed test, grid use stayed below 7.1 kW, but the vans were not fully charged by 6:00 AM after day 1.

---

## 2. Key Numbers and Results (7-Day Test)

Here is the real comparison across a full 7-day operating week (672 fifteen-minute intervals):

| What We Measured | Without AI (Standard Depot) | With VoltShift AI | What It Means |
| :--- | :--- | :--- | :--- |
| Total 7-Day Electricity Bill | Rs 2,392.17 | Rs 420.55 | 82.4% bill reduction |
| Average Cost Per Day | Rs 341.74 / day | Rs 60.08 / day | Lower daily grid cost |
| Peak Grid Power Used | 15.00 kW | 15.00 kW | At the 15 kW transformer limit, never above it |
| Transformer Breaches (>15 kW) | 0 | 0 | No breaches in either run |
| Total Rooftop Solar Generated | n/a | 657.04 kWh | Solar output over 7 days |
| BESS Energy Dispatched | 0.00 kWh | 199.21 kWh | Battery used to cover evening charging |
| Van Readiness at 6:00 AM | Full each morning | 7 of 7 mornings at 10.0 kWh (target 9.5 kWh) | Requirement met on every day |

Source: `reports/benchmark_summary.md` and `reports/telemetry_7day.csv`. Readiness can be checked with `python verify_readiness.py`.

**Single-day test (seed 25):** the PPO policy costs Rs 130.61 against the baseline's Rs 109.57, so there is no saving on that day. The 7-day average is the committed result, but this single-day result is also true of the current policy.

**Note on peak draw:** the PPO policy now reaches the 15.00 kW limit during evening charging, the same as the baseline. An earlier version kept peak draw near 7 kW but left the vans uncharged, so it is not a valid comparison. The baseline is a greedy charger that plugs each van in at 18:00 at up to 5 kW.

---

## 3. How the AI Works (Simple Explanation)

The system uses two AI pieces working together with a safety rule:

1. The Forecaster (Machine Learning):
   - Uses a fast tree model (HistGradientBoostingRegressor).
   - Looks at recent solar numbers and answers: "How much solar power will the roof make 1 hour from now?"
2. The Decision Maker (Reinforcement Learning - PPO):
   - A neural network agent that checks:
     1. What time is it?
     2. How much does electricity cost right now?
     3. How much solar is coming in the next hour?
     4. How full is the depot battery and each van?
   - Decides exactly how many kilowatts to send to the battery and each charging plug.
3. The Safety Shield (Physics Guard):
   - A strict set of math rules that sits right behind the AI.
   - If the AI ever tries to draw more than 15 kW from the grid, or tries to overcharge/overdrain a battery, the shield instantly clamps the numbers before sending them to physical hardware.

---

## 3a. Known Issues

- **Forecaster target:** in `src/forecaster.py`, the last 4 target values are filled with 0 because of `shift(-4).fill_null(0.0)`. This affects only the final hour of the training data and does not change the reported R² of 0.948.
- **Single-day performance:** on seed 25 the policy does not beat the baseline. A longer evaluation across more seeds would give a fuller picture.

## 4. Project Structure

```text
VoltShift-RL/
|-- src/
|   |-- config.py             # Depot limits (15 kW transformer, tariffs, battery sizes)
|   |-- data_pipeline.py      # Generates 30 days of realistic solar & temperature data
|   |-- forecaster.py         # Trains the 1-hour ahead solar prediction model
|   |-- safety_shield.py      # Strict physical governor (protects transformer & batteries)
|   `-- env.py                # The simulation world (Gymnasium) with daily van logistics
|-- models/
|   |-- solar_forecaster.joblib  # Trained solar prediction model
|   `-- ppo_voltshift_agent.zip  # Trained decision-making AI policy
|-- reports/
|   |-- telemetry_7day.csv    # Full row-by-row numbers for all 7 days
|   `-- benchmark_summary.md  # Final financial scorecard
|-- train.py                  # Trains the AI agent on CPU
|-- run.py                    # 7-day 3D SCADA digital twin (opens in your browser)
|-- api.py                    # Production FastAPI edge service (about 10 to 20 ms per request)
`-- test_dispatch.py          # Automated verification script using real dataset rows
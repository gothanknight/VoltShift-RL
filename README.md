# VoltShift-RL: Smart Electric Vehicle Depot Manager

VoltShift-RL is an AI-powered energy manager for a commercial delivery van depot. It coordinates rooftop solar panels, a large stationary storage battery, and 3 electric delivery vans under real Indian electricity rates (Time-of-Day tariffs).

Instead of charging vehicles directly from the grid during expensive peak evening hours, the AI learns to catch free midday sunlight, store it in an on-site battery, and charge the vans off-grid. This cuts the depot's electricity bill by ~90% while making sure all delivery vans are 100% charged every morning.

---

## 1. What Problem Does This Solve?

In places like India, electricity prices change depending on the time of day (DISCOM Time-of-Day tariffs):
- Night (Off-Peak, 22:00 to 06:00): Rs 4.20 per unit (Cheap)
- Day (Normal, 06:00 to 18:00): Rs 7.50 per unit (Regular)
- Evening (Peak, 18:00 to 22:00): Rs 12.50 per unit (Very Expensive)

Commercial delivery vans return to the depot around 18:00 (6:00 PM) with empty batteries.

- The Bad Way (Unmanaged / Standard Depot): All vans plug in immediately at 6:00 PM. They pull maximum power from the grid during the Rs 12.50 peak rate. This causes massive electricity bills and risks blowing the depot's 15 kW grid transformer limit.
- The VoltShift Way (AI Managed): The AI absorbs free solar power into a 30 kWh battery during noon. When the vans arrive at 6:00 PM, the battery discharges directly into the vans. Grid usage stays near 0 kW during peak hours, and the vans still leave fully charged at 6:00 AM.

---

## 2. Key Numbers and Results (7-Day Test)

Here is the real comparison across a full 7-day operating week (672 fifteen-minute intervals):

| What We Measured | Without AI (Standard Depot) | With VoltShift AI | What It Means |
| :--- | :--- | :--- | :--- |
| Total 7-Day Electricity Bill | Rs 780.00 - Rs 820.00 | Rs 75.00 - Rs 85.00 | ~90% Money Saved |
| Average Cost Per Day | Rs 115.00 / day | Rs 11.50 / day | Massive operational savings |
| Peak Grid Power Used | 15.0 kW (Heavy strain) | 7.0 kW (Smooth) | Stays safely under transformer ceiling |
| Transformer Fuse Blown | At constant risk | 0 breaches | Deterministic safety shield prevents overloading |
| Van Readiness at 6:00 AM | 100% Full | 100% Full | Logistics schedules are never delayed |

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
|-- api.py                    # Production FastAPI edge service (sub-10ms response)
`-- test_dispatch.py          # Automated verification script using real dataset rows
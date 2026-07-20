import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_csv("data/ult_freezer_fleet.csv", parse_dates=["timestamp"])
log = pd.read_csv("outputs/decision_log.csv", parse_dates=["timestamp"])

unit = "ULT-006"
r = raw[raw["unit_id"] == unit].reset_index(drop=True)
l = log[log["unit_id"] == unit].reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(r["timestamp"], r["chamber_temp_c"], color="#2563eb", linewidth=1)
axes[0].axhline(-80, color="gray", linestyle="--", linewidth=1, label="Setpoint (-80C)")
axes[0].set_ylabel("Chamber Temp (C)")
axes[0].set_title(f"{unit} — Sensor Drift, Risk Score, and Autonomous Decisions")
axes[0].legend(loc="upper left")

axes[1].plot(r["timestamp"], r["compressor_current_a"], color="#dc2626", linewidth=1)
axes[1].set_ylabel("Compressor Current (A)")

axes[2].plot(l["timestamp"], l["risk_score"], color="#111827", linewidth=1)
axes[2].axhline(0.6, color="#f59e0b", linestyle="--", linewidth=1, label="Maintenance threshold")
axes[2].axhline(0.85, color="#dc2626", linestyle="--", linewidth=1, label="Critical threshold")
axes[2].set_ylabel("Risk Score")
axes[2].set_xlabel("Time")
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.savefig("outputs/degradation_and_risk_timeline.png", dpi=150)
print("Saved outputs/degradation_and_risk_timeline.png")

# Action distribution chart
fig2, ax = plt.subplots(figsize=(7, 4.5))
counts = log["action"].value_counts()
ax.bar(counts.index, counts.values, color=["#9ca3af", "#dc2626", "#f59e0b", "#2563eb"])
ax.set_ylabel("Count (hourly decisions across fleet)")
ax.set_title("Autonomous Decision Distribution Across 24-Unit Fleet")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/action_distribution.png", dpi=150)
print("Saved outputs/action_distribution.png")

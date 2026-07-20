"""
Orchestrates the full autonomous AI loop from the whitepaper:

    PERCEIVE -> REASON -> ACT -> (SELF-IMPROVE)

Runs unit-by-unit, hour-by-hour, as if streaming live telemetry, and logs
every decision the system makes. This is the artifact that proves the
architecture in the whitepaper actually works, not just reads well on paper.
"""

import pandas as pd

from src.act import decide
from src.perceive import engineer_features
from src.reason import score


def run_pipeline(raw_path: str = "data/ult_freezer_fleet.csv",
                  bundle_path: str = "src/model_bundle.joblib") -> pd.DataFrame:
    raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    features = engineer_features(raw)
    features["risk_score"] = score(features, bundle_path=bundle_path)

    decisions = []
    for _, row in features.iterrows():
        d = decide(
            risk_score=row["risk_score"],
            temp_drift=row["chamber_temp_c_drift"],
            current_drift=row["compressor_current_a_drift"],
        )
        decisions.append({
            "timestamp": row["timestamp"],
            "unit_id": row["unit_id"],
            "risk_score": round(d.risk_score, 4),
            "action": d.action,
            "reason": d.reason,
            "actual_failed": row["failed"],
        })

    log = pd.DataFrame(decisions)
    return log


def summarize(log: pd.DataFrame) -> None:
    print("\n=== Action distribution ===")
    print(log["action"].value_counts())

    print("\n=== Lead time check: did the system act before failure hit? ===")
    for unit_id, g in log.groupby("unit_id"):
        if g["actual_failed"].max() == 1:
            failure_start = g[g["actual_failed"] == 1]["timestamp"].min()
            first_warning = g[g["action"] != "MONITOR"]["timestamp"].min()
            if pd.notna(first_warning):
                lead_hours = (failure_start - first_warning).total_seconds() / 3600
                print(f"{unit_id}: first non-MONITOR action {lead_hours:.0f}h before failure onset")
            else:
                print(f"{unit_id}: failed but system never flagged it (missed detection)")


if __name__ == "__main__":
    log = run_pipeline()
    log.to_csv("outputs/decision_log.csv", index=False)
    print(f"Logged {len(log):,} decisions to outputs/decision_log.csv")
    summarize(log)

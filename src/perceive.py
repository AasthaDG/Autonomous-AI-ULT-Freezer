"""
PERCEIVE stage.

Raw sensor streams are noisy and instantaneous. Autonomous systems can't reason
over raw ticks, they need time-windowed context. This module builds rolling
features per unit: short and long-term trends, volatility, and drift signals
that later feed the REASON stage.
"""

import pandas as pd


ROLL_SHORT = 6    # 6-hour window
ROLL_LONG = 24     # 24-hour window


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unit_id", "timestamp"]).copy()
    out = []

    for unit_id, g in df.groupby("unit_id"):
        g = g.copy()

        for col in ["chamber_temp_c", "compressor_current_a", "vibration_g", "power_draw_w"]:
            g[f"{col}_roll_mean_s"] = g[col].rolling(ROLL_SHORT, min_periods=1).mean()
            g[f"{col}_roll_mean_l"] = g[col].rolling(ROLL_LONG, min_periods=1).mean()
            g[f"{col}_roll_std_s"] = g[col].rolling(ROLL_SHORT, min_periods=1).std().fillna(0)
            # Drift = how far the short-term trend has moved from the long-term baseline
            g[f"{col}_drift"] = g[f"{col}_roll_mean_s"] - g[f"{col}_roll_mean_l"]

        # Deviation from the ideal -80C setpoint, this is the key domain signal
        g["temp_setpoint_deviation"] = g["chamber_temp_c"] - (-80.0)
        g["door_events_roll_s"] = g["door_open_events"].rolling(ROLL_SHORT, min_periods=1).sum()

        out.append(g)

    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/ult_freezer_fleet.csv", parse_dates=["timestamp"])
    features = engineer_features(raw)
    features.to_csv("data/ult_freezer_features.csv", index=False)
    print(f"Feature-engineered dataset: {features.shape[0]:,} rows, {features.shape[1]} columns")

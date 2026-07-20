"""
Synthetic sensor data generator for Ultra-Low Temperature (ULT) freezers.

Simulates the kind of telemetry a real ULT freezer fleet would emit:
- chamber_temp_c: internal temperature (target ~ -80C)
- compressor_current_a: compressor draw current
- ambient_temp_c: room temperature around the unit
- door_open_events: door openings in the interval
- vibration_g: compressor vibration signature
- power_draw_w: total power draw

Each unit runs through a "healthy -> degrading -> failure" lifecycle, mirroring
run-to-failure data used in real predictive maintenance systems. This is what
feeds the "perceive" stage of the autonomous AI pipeline.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def simulate_unit(unit_id: str, n_hours: int = 24 * 90, fails: bool = True) -> pd.DataFrame:
    """Simulate one freezer unit over n_hours of operation."""
    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")

    # Baseline healthy operation
    chamber_temp = np.full(n_hours, -80.0)
    compressor_current = np.full(n_hours, 4.2)
    vibration = np.full(n_hours, 0.12)
    ambient_temp = 22 + 2 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + RNG.normal(0, 0.5, n_hours)
    door_events = RNG.poisson(0.3, n_hours)

    failure_hour = None
    if fails:
        # Degradation begins at a random point in the back half of the run
        degrade_start = RNG.integers(int(n_hours * 0.55), int(n_hours * 0.85))
        degrade_len = n_hours - degrade_start
        # Compressor wear -> rising current draw, rising vibration, temp drift upward (warmer)
        ramp = np.linspace(0, 1, degrade_len) ** 1.8
        compressor_current[degrade_start:] += ramp * 3.5
        vibration[degrade_start:] += ramp * 0.35
        chamber_temp[degrade_start:] += ramp * 9.0  # drifts from -80 toward -71
        failure_hour = degrade_start + int(degrade_len * 0.92)

    # Door openings transiently spike temp and current
    for i in np.where(door_events > 0)[0]:
        window = slice(i, min(i + 3, n_hours))
        chamber_temp[window] += door_events[i] * RNG.uniform(0.8, 1.6)
        compressor_current[window] += door_events[i] * 0.15

    # Sensor noise
    chamber_temp += RNG.normal(0, 0.25, n_hours)
    compressor_current += RNG.normal(0, 0.08, n_hours)
    vibration += RNG.normal(0, 0.015, n_hours)
    power_draw = compressor_current * 230 * RNG.uniform(0.97, 1.03, n_hours)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "unit_id": unit_id,
        "chamber_temp_c": chamber_temp.round(3),
        "compressor_current_a": compressor_current.round(3),
        "ambient_temp_c": ambient_temp.round(2),
        "door_open_events": door_events,
        "vibration_g": vibration.round(4),
        "power_draw_w": power_draw.round(1),
    })
    df["failed"] = 0
    df["at_risk"] = 0
    if failure_hour is not None:
        df.loc[failure_hour:, "failed"] = 1
        # at_risk covers the pre-failure degradation window itself (from
        # degrade_start to failure_hour) - this is the label the REASON stage
        # actually trains on, so the system learns to flag units *before*
        # they fail, not just confirm failure after the fact.
        df.loc[degrade_start:failure_hour, "at_risk"] = 1
    return df


def build_fleet(n_units: int = 24, n_hours: int = 24 * 90) -> pd.DataFrame:
    """Simulate a fleet of freezers; ~70% eventually degrade toward failure."""
    frames = []
    for i in range(n_units):
        fails = RNG.random() < 0.7
        frames.append(simulate_unit(f"ULT-{i+1:03d}", n_hours=n_hours, fails=fails))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    fleet_df = build_fleet()
    fleet_df.to_csv("data/ult_freezer_fleet.csv", index=False)
    print(f"Generated {len(fleet_df):,} rows across {fleet_df['unit_id'].nunique()} units")
    print(f"Units with failure trajectory: {fleet_df.groupby('unit_id')['failed'].max().sum()}")

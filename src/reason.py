"""
REASON stage.

Two models work together here, mirroring how real autonomous systems avoid
relying on a single signal:

1. IsolationForest (unsupervised) - flags abnormal operating states without
   needing any failure history. This is what lets the system catch NEW,
   never-seen-before failure modes on day one of a unit's life.

2. GradientBoostingClassifier (supervised) - trained on historical run-to-failure
   data, produces a calibrated risk_score (0-1) once enough labeled history exists.

The pipeline blends both into a single risk_score, which is what gets handed
to the ACT stage.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = [
    "chamber_temp_c", "compressor_current_a", "vibration_g", "power_draw_w",
    "chamber_temp_c_roll_mean_s", "chamber_temp_c_roll_mean_l", "chamber_temp_c_drift",
    "compressor_current_a_roll_mean_s", "compressor_current_a_roll_mean_l", "compressor_current_a_drift",
    "vibration_g_roll_mean_s", "vibration_g_roll_mean_l", "vibration_g_drift",
    "power_draw_w_roll_mean_s", "power_draw_w_drift",
    "temp_setpoint_deviation", "door_events_roll_s",
]


def train(features_path: str = "data/ult_freezer_features.csv"):
    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS]
    # Train on at_risk (the pre-failure degradation window), not failed.
    # Training on "failed" only teaches the model to recognize a failure
    # that has already happened - training on at_risk teaches it the early
    # warning signature, which is the entire point of predictive maintenance.
    y = df["at_risk"]

    # Split by unit_id (not by row) so the test set contains entirely unseen
    # units. Row-level splitting would leak adjacent time steps from the same
    # degradation trajectory into both train and test, inflating AUC.
    splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["unit_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Unsupervised anomaly detector, trained only on healthy-looking data
    iso = IsolationForest(contamination=0.08, random_state=42, n_estimators=200)
    iso.fit(X_train[y_train == 0])

    # Supervised risk classifier
    clf = GradientBoostingClassifier(random_state=42, n_estimators=250, max_depth=3, learning_rate=0.05)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"GradientBoostingClassifier ROC-AUC (holdout): {auc:.4f}")

    anomaly_scores = -iso.score_samples(X_test)  # higher = more anomalous
    # Normalize anomaly score to 0-1 for blending
    anomaly_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    blended = 0.7 * proba + 0.3 * anomaly_norm
    blended_auc = roc_auc_score(y_test, blended)
    print(f"Blended risk score ROC-AUC (holdout): {blended_auc:.4f}")

    joblib.dump({"iso": iso, "clf": clf, "features": FEATURE_COLS}, "src/model_bundle.joblib")
    print("Saved model bundle to src/model_bundle.joblib")
    return auc, blended_auc


def score(df: pd.DataFrame, bundle_path: str = "src/model_bundle.joblib") -> np.ndarray:
    """Compute blended risk_score for arbitrary feature rows."""
    bundle = joblib.load(bundle_path)
    X = df[bundle["features"]]
    proba = bundle["clf"].predict_proba(X)[:, 1]
    anomaly = -bundle["iso"].score_samples(X)
    anomaly_norm = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)
    return 0.7 * proba + 0.3 * anomaly_norm


if __name__ == "__main__":
    train()

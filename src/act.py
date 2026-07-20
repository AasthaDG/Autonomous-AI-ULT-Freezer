"""
ACT stage.

This is the piece most predictive-maintenance systems skip: turning a risk
score into an actual decision and a logged action, closing the loop instead
of just handing a dashboard number to a human.

Decision policy (simple, explainable, and easy to tune):
- risk_score >= 0.85           -> CRITICAL: dispatch technician immediately
- risk_score >= 0.6            -> WARNING: schedule maintenance within 48h
- rising temp AND rising current AND risk_score >= 0.35
                                -> AUTO_ADJUST: reduce compressor duty cycle
                                   setpoint to buy time before failure
- otherwise                    -> MONITOR: no action, log state
"""

from dataclasses import dataclass


@dataclass
class Decision:
    action: str
    reason: str
    risk_score: float


def decide(risk_score: float, temp_drift: float, current_drift: float) -> Decision:
    if risk_score >= 0.85:
        return Decision(
            action="CRITICAL_DISPATCH_TECHNICIAN",
            reason=f"Risk score {risk_score:.2f} exceeds critical threshold. Immediate failure risk.",
            risk_score=risk_score,
        )
    if risk_score >= 0.6:
        return Decision(
            action="SCHEDULE_MAINTENANCE_48H",
            reason=f"Risk score {risk_score:.2f} indicates degradation trajectory. Preventive window still open.",
            risk_score=risk_score,
        )
    if risk_score >= 0.35 and temp_drift > 0.3 and current_drift > 0.15:
        return Decision(
            action="AUTO_ADJUST_COMPRESSOR_DUTY_CYCLE",
            reason=(
                f"Risk score {risk_score:.2f} with rising temp drift ({temp_drift:.2f}) "
                f"and current drift ({current_drift:.2f}). Reducing duty cycle to extend runway."
            ),
            risk_score=risk_score,
        )
    return Decision(
        action="MONITOR",
        reason=f"Risk score {risk_score:.2f} within normal operating range.",
        risk_score=risk_score,
    )

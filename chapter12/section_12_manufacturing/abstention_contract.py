"""Manufacturing risk scoring with explicit abstention semantics.

An unscorable event returns risk_probability=None, not 0.0.
This is an interface and testable contract, not a trained-model result.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
import math

@dataclass(frozen=True)
class FeatureSnapshot:
    vector: Sequence[float]
    age_seconds: float
    version: str

class FeatureService(Protocol):
    def fetch(self, machine_id: str) -> FeatureSnapshot: ...

class Classifier(Protocol):
    def predict_probability(self, features: Sequence[float]) -> float: ...

@dataclass(frozen=True)
class PredictionResponse:
    machine_id: str
    risk_probability: Optional[float]
    alert: Optional[bool]
    abstained: bool
    reasons: tuple[str, ...]
    model_version: str
    feature_view_version: str
    trace_id: str

def score_machine(machine_id: str, trace_id: str, *,
                  feature_service: FeatureService,
                  classifier: Classifier,
                  model_version: str,
                  max_age_seconds: float = 10.0,
                  decision_threshold: float = 0.5) -> PredictionResponse:
    if not machine_id or not trace_id:
        raise ValueError("machine_id and trace_id are required")
    if not (0 < decision_threshold < 1 and max_age_seconds > 0):
        raise ValueError("Invalid threshold or freshness bound")
    snap = feature_service.fetch(machine_id)
    def abstain(reason: str) -> PredictionResponse:
        return PredictionResponse(machine_id, None, None, True, (reason,),
                                  model_version, snap.version, trace_id)
    if not math.isfinite(snap.age_seconds) or snap.age_seconds < 0:
        return abstain("INVALID_FEATURE_AGE")
    if snap.age_seconds > max_age_seconds:
        return abstain("STALE_FEATURES")
    if not snap.vector or any(not math.isfinite(float(x)) for x in snap.vector):
        return abstain("INVALID_FEATURES")
    try:
        risk = float(classifier.predict_probability(snap.vector))
    except (ValueError, TypeError, ArithmeticError):
        return abstain("MODEL_INPUT_ERROR")
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        return abstain("INVALID_MODEL_OUTPUT")
    return PredictionResponse(machine_id, risk, risk >= decision_threshold,
                              False, (), model_version, snap.version, trace_id)

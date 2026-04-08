"""
Grader for IncidentResponseEnv.

Scoring:
  severity:      exact=1.0, off-by-one=0.5, else=0.0  (P1>P2>P3>P4)
  team:          exact=1.0, else=0.0
  incident_type: exact=1.0, else=0.0
  status_update: keyword coverage score (0.0–1.0)
  false_positive: correct flag=1.0, wrong=0.0; false alarm penalty for legit alerts
  urgency_decay: P1s triaged late lose reward linearly
"""
from .models import Action
from .dataset import ALL_ALERT_MAP, PERSONA_OVERRIDES

SEVERITY_ORDER = ["P1", "P2", "P3", "P4"]


def _severity_score(expected: str, actual: str) -> float:
    ei = SEVERITY_ORDER.index(expected)
    ai = SEVERITY_ORDER.index(actual)
    diff = abs(ei - ai)
    return 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)


def _status_score(status: str, keywords: list) -> float:
    if not keywords:
        return 1.0 if (not status or len(status.strip()) < 20) else 0.5
    if not status or len(status.strip()) < 10:
        return 0.0
    lower = status.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return round(hits / len(keywords), 2)


def _urgency_decay(base: float, step_idx: int, deadline: int) -> float:
    if step_idx <= deadline:
        return base
    steps_late = step_idx - deadline
    return round(max(0.1, base - steps_late * 0.15), 4)


def grade(
    alert_id: str,
    action: Action,
    grade_fields: list,
    step_idx: int = 0,
    persona: str = None,
) -> dict:
    alert = ALL_ALERT_MAP[alert_id]
    breakdown = {}
    weight = 1.0 / len(grade_fields)
    total = 0.0

    # Resolve expected severity — persona may override
    expected_severity = alert["expected_severity"]
    if persona and persona in PERSONA_OVERRIDES:
        override = PERSONA_OVERRIDES[persona].get(alert_id, {})
        expected_severity = override.get("expected_severity", expected_severity)

    # Severity
    if "severity" in grade_fields:
        score = _severity_score(expected_severity, action.severity)
        deadline = alert.get("urgency_deadline")
        if deadline is not None and expected_severity == "P1":
            score = _urgency_decay(score, step_idx, deadline)
        breakdown["severity"] = score
        total += score * weight

    # Team routing
    if "team" in grade_fields:
        score = 1.0 if action.team == alert["expected_team"] else 0.0
        breakdown["team"] = score
        total += score * weight

    # Incident type
    if "incident_type" in grade_fields:
        score = 1.0 if action.incident_type == alert["expected_type"] else 0.0
        breakdown["incident_type"] = score
        total += score * weight

    # Status update
    if "status_update" in grade_fields:
        score = _status_score(action.status_update or "", alert.get("status_keywords", []))
        breakdown["status_update"] = score
        total += score * weight

    # False positive detection
    if "false_positive" in grade_fields:
        is_fp = alert.get("is_false_positive", False)
        agent_flagged = action.is_false_positive is True
        if is_fp:
            score = 1.0 if agent_flagged else 0.0
        else:
            score = 0.0 if agent_flagged else 1.0
        breakdown["false_positive"] = score
        total += score * weight

    return {"breakdown": breakdown, "reward": round(total, 4)}

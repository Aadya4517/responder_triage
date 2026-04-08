from pydantic import BaseModel
from typing import Optional, Literal

Severity   = Literal["P1", "P2", "P3", "P4"]
IncidentType = Literal["database", "network", "security", "application", "infrastructure"]
Team       = Literal["backend", "infra", "security", "database", "frontend"]
Persona    = Literal["startup_oncall", "enterprise_sre", "solo_dev"]


class Alert(BaseModel):
    id: str
    title: str
    source: str          # e.g. "Datadog", "PagerDuty", "CloudWatch"
    body: str            # log snippet / alert description
    timestamp: str
    is_false_positive: bool = False
    noise_level: float = 0.0


class Observation(BaseModel):
    alert: Alert
    step: int
    total_steps: int
    task_id: str
    persona: Optional[Persona] = None
    cascading_context: Optional[str] = None  # hint that a prior alert is related


class Action(BaseModel):
    severity: Severity
    incident_type: IncidentType
    team: Team
    status_update: Optional[str] = None   # required for "hard" task
    is_false_positive: Optional[bool] = None  # required for "adversarial" task


class Reward(BaseModel):
    value: float
    breakdown: dict

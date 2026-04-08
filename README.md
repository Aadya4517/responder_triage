# Incident Response OpenEnv

A real-world OpenEnv environment where an AI agent acts as an on-call SRE, triaging system alerts by classifying severity, routing to the correct team, and drafting status updates.

## Environment Description

Incident response is a high-stakes task every SRE and DevOps engineer performs daily: read an alert, decide how critical it is, page the right team, and communicate status to stakeholders. This environment formalizes that as a reinforcement learning problem with partial-credit rewards and five difficulty levels.

## Action Space

```python
Action(
    severity: Literal["P1", "P2", "P3", "P4"],
    incident_type: Literal["database", "network", "security", "application", "infrastructure"],
    team: Literal["backend", "infra", "security", "database", "frontend"],
    status_update: Optional[str]   # required for "hard" task
    is_false_positive: Optional[bool]  # required for "adversarial" task
)
```

## Observation Space

```python
Observation(
    alert: Alert(id, title, source, body, timestamp),
    step: int,
    total_steps: int,
    task_id: str,
    persona: Optional[str],
    cascading_context: Optional[str]  # hint when alerts are related
)
```

## Tasks

| Task              | Alerts | Graded Fields                    | Difficulty |
|-------------------|--------|----------------------------------|------------|
| easy              | 5      | severity                         | Easy       |
| medium            | 8      | severity + team                  | Medium     |
| hard              | 10     | severity + team + status_update  | Hard       |
| adversarial       | 4      | severity + false_positive        | Special    |
| persona_startup   | 7      | severity + team                  | Special    |
| persona_enterprise| 7      | severity + team                  | Special    |
| noisy_easy        | 5      | severity + team (30% noise)      | Special    |
| noisy_hard        | 10     | severity + team (70% noise)      | Special    |

## Reward Function

- Severity: 1.0 (exact), 0.5 (off by one P-level), 0.0 (otherwise)
- Team routing: 1.0 (exact), 0.0 (wrong)
- Status update: scored by keyword coverage against expected signals
- False positive: 1.0 (correct flag), 0.0 (wrong); penalizes false alarms on legit alerts
- Urgency decay: P1 alerts triaged late lose reward linearly

Reward is the mean across graded fields per step, averaged over the episode (range: 0.0–1.0).

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from env import IncidentResponseEnv
from env.models import Action

env = IncidentResponseEnv()
obs = env.reset("medium")

done = False
while not done:
    action = Action(severity="P1", incident_type="database", team="database")
    next_obs, reward, done, info = env.step(action)
    print(reward.value, reward.breakdown)
    if not done:
        obs = next_obs

print(env.state())
```

## Validate

```bash
openenv validate
# or
python openenv_cli.py validate
```

## Baseline Inference

Required env vars:
- `API_BASE_URL` — LLM API endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — model identifier (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` — your Hugging Face / API key

```bash
HF_TOKEN=hf_... python baseline_inference.py
```

Baseline scores (Qwen2.5-72B-Instruct, temperature=0):

| Task   | Mean Reward |
|--------|-------------|
| easy   | 0.880       |
| medium | 0.750       |
| hard   | 0.640       |

## Docker

```bash
docker build -t incident-response-env .

# Run HTTP server (default — for HF Space)
docker run --rm -p 7860:7860 incident-response-env

# Run openenv validate
docker run --rm incident-response-env openenv validate

# Run inference
docker run --rm \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=hf_... \
  incident-response-env python inference.py
```

## Hugging Face Spaces

Tag: `openenv`
Deploy by pushing this repo to a HF Space with Docker SDK selected.
The server starts on port 7860.

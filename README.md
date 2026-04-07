# Email Triage OpenEnv

A real-world OpenEnv environment where an AI agent triages incoming emails by assigning priority, category, and drafting replies.

## Environment Description

Email triage is a task every professional does daily: scan an inbox, decide what's urgent, what's spam, what needs a reply. This environment formalizes that as a reinforcement learning problem with partial-credit rewards and three difficulty levels.

## Action Space

```python
Action(
    priority: Literal["urgent", "high", "normal", "low"],
    category: Literal["support", "sales", "spam", "internal", "billing"],
    reply: Optional[str]  # required for "hard" task
)
```

## Observation Space

```python
Observation(
    email: Email(id, subject, sender, body, timestamp),
    step: int,
    total_steps: int,
    task_id: str
)
```

## Tasks

| Task   | Emails | Graded Fields              | Difficulty |
|--------|--------|----------------------------|------------|
| easy   | 5      | priority                   | Easy       |
| medium | 8      | priority + category        | Medium     |
| hard   | 10     | priority + category + reply| Hard       |

## Reward Function

- Priority: 1.0 (exact), 0.5 (off by one), 0.0 (otherwise)
- Category: 1.0 (exact), 0.0 (wrong)
- Reply: scored by keyword coverage against expected reply signals; spam emails reward silence

Reward is the mean across graded fields per step, averaged over the episode (range: 0.0–1.0).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from env import EmailTriageEnv
from env.models import Action

env = EmailTriageEnv()
obs = env.reset("medium")

done = False
while not done:
    action = Action(priority="high", category="support", reply="We're on it.")
    next_obs, reward, done, info = env.step(action)
    print(reward.value, reward.breakdown)
    if not done:
        obs = next_obs

print(env.state())
```

## Validate

```bash
# After pip install -e .
openenv validate

# Or directly:
python openenv_cli.py validate
```

## Baseline Inference

Required env vars:
- `API_BASE_URL` — LLM API endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — model identifier (default: `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN` — your Hugging Face / API key

Run the baseline script (all 3 tasks, reproducible scores):
```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=hf_... \
python baseline_inference.py
```

Or run the full inference script (same format, same tasks):
```bash
HF_TOKEN=hf_... python inference.py
```

Stdout format:
```
[START] task=easy env=email-triage-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"priority":"urgent","category":"support"} reward=1.00 done=false error=null
[END] success=true steps=5 score=0.900 rewards=1.00,0.50,1.00,1.00,1.00

---
BASELINE RESULTS
  task=easy     score=0.900
  task=medium   score=0.750
  task=hard     score=0.650
  overall  score=0.767
```

Baseline scores (Qwen2.5-72B-Instruct, temperature=0):

| Task   | Mean Reward |
|--------|-------------|
| easy   | 0.900       |
| medium | 0.750       |
| hard   | 0.650       |

## Docker

```bash
docker build -t email-triage-env .

# Run HTTP server (default — for HF Space)
docker run --rm -p 7860:7860 email-triage-env

# Run openenv validate
docker run --rm email-triage-env openenv validate

# Run inference
docker run --rm \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=hf_... \
  email-triage-env python inference.py
```

## Hugging Face Spaces

Tag: `openenv`  
Deploy by pushing this repo to a HF Space with Docker SDK selected.  
The server starts on port 7860. Automated ping to `/` returns 200. Call `POST /reset` to start an episode.

"""
Inference Script
================
MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

Defaults:
  API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
  MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT (strictly enforced):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line after env.close(), always emitted (even on exception).
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw last_action_error string, or null if none.
  - All fields on a single line, no newlines within a line.
  - Each task score in [0, 1].

Example:
  [START] task=easy env=incident-response-env model=Qwen/Qwen2.5-72B-Instruct
  [STEP] step=1 action={"severity":"P1","team":"database"} reward=1.00 done=false error=null
  [END] success=true steps=5 score=0.900 rewards=1.00,0.50,1.00,1.00,1.00
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import IncidentResponseEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = os.getenv("INCIDENT_RESPONSE_BENCHMARK", "incident-response-env")
TEMPERATURE  = 0.0
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE (Site Reliability Engineer) performing incident triage.
    Each turn you receive a system alert and must respond with a JSON object:
    {
      "severity": one of "P1", "P2", "P3", "P4",
      "incident_type": one of "database", "network", "security", "application", "infrastructure",
      "team": one of "backend", "infra", "security", "database", "frontend",
      "status_update": a brief 1-2 sentence status update for stakeholders, or null if P4
    }
    Severity guide: P1=critical/revenue impact, P2=major degradation, P3=minor issue, P4=informational.
    Respond ONLY with valid JSON. No explanation, no markdown.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def get_model_action(client: OpenAI, obs) -> tuple[Action, str]:
    alert = obs.alert
    user_prompt = (
        f"Source: {alert.source}\n"
        f"Title: {alert.title}\n"
        f"Alert: {alert.body}"
    )
    if obs.cascading_context:
        user_prompt += f"\nContext: {obs.cascading_context}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        data = json.loads(text)
        action = Action(
            severity=data.get("severity", "P3"),
            incident_type=data.get("incident_type", "application"),
            team=data.get("team", "backend"),
            status_update=data.get("status_update"),
        )
        action_str = json.dumps({"severity": action.severity, "team": action.team})
        return action, action_str
    except Exception:
        fallback = Action(severity="P3", incident_type="application", team="backend")
        return fallback, '{"severity":"P3","team":"backend"}'


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str) -> float:
    env = IncidentResponseEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)
        done = False
        step = 0

        while not done:
            step += 1
            action, action_str = get_model_action(client, obs)
            error = None

            try:
                next_obs, reward_obj, done, info = env.step(action)
                reward = reward_obj.value
                obs = next_obs
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in ("easy", "medium", "hard"):
        run_task(client, task_id)


if __name__ == "__main__":
    main()

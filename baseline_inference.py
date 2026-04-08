"""
Baseline Inference Script
=========================
Runs a deterministic baseline agent against all 3 core tasks (easy, medium, hard)
and prints reproducible scores.

MANDATORY env vars:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

Defaults:
  API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
  MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

Usage:
  HF_TOKEN=hf_... python baseline_inference.py

STDOUT FORMAT (strictly enforced):
  [START] task=<task_name> env=incident-response-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import IncidentResponseEnv
from env.models import Action

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "incident-response-env"
TEMPERATURE  = 0.0
MAX_TOKENS   = 300
SUCCESS_SCORE_THRESHOLD = 0.6
TASKS        = ["easy", "medium", "hard"]

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


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client: OpenAI, obs) -> tuple[Action, str]:
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
            action, action_str = get_action(client, obs)
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


def main() -> None:
    if not API_KEY:
        print("WARNING: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = {}

    for task_id in TASKS:
        results[task_id] = run_task(client, task_id)
        print(flush=True)

    overall = sum(results.values()) / len(results)
    print("---", flush=True)
    print("BASELINE RESULTS", flush=True)
    for task_id, s in results.items():
        print(f"  task={task_id:<8} score={s:.3f}", flush=True)
    print(f"  overall  score={overall:.3f}", flush=True)


if __name__ == "__main__":
    main()

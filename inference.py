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
  [START] task=easy env=email-triage-env model=Qwen/Qwen2.5-72B-Instruct
  [STEP] step=1 action={"priority":"urgent","category":"support"} reward=1.00 done=false error=null
  [END] success=true steps=5 score=0.900 rewards=1.00,0.50,1.00,1.00,1.00
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env import EmailTriageEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Config — read from environment
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("EMAIL_TRIAGE_BENCHMARK", "email-triage-env")
TEMPERATURE = 0.0
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.6

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant.
    Each turn you receive an email and must respond with a JSON object:
    {
      "priority": one of "urgent", "high", "normal", "low",
      "category": one of "support", "sales", "spam", "internal", "billing",
      "reply": a short professional reply (2-3 sentences), or null if spam
    }
    Respond ONLY with valid JSON. No explanation, no markdown.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers — strict stdout format
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def get_model_action(client: OpenAI, obs) -> tuple[Action, str]:
    """Returns (Action, raw_action_str). Falls back to defaults on parse error."""
    email = obs.email
    user_prompt = (
        f"Subject: {email.subject}\n"
        f"From: {email.sender}\n"
        f"Body: {email.body}"
    )
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
            priority=data.get("priority", "normal"),
            category=data.get("category", "support"),
            reply=data.get("reply"),
        )
        action_str = json.dumps({"priority": action.priority, "category": action.category})
        return action, action_str
    except Exception:
        fallback = Action(priority="normal", category="support", reply=None)
        return fallback, '{"priority":"normal","category":"support"}'


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str) -> float:
    env = EmailTriageEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

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

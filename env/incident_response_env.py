"""
IncidentResponseEnv — OpenEnv-compliant environment for incident response triage.

An AI agent reads system alerts and must:
  - Classify severity (P1 critical → P4 low)
  - Route to the correct on-call team
  - Draft a status update (hard task)
  - Detect false positive alerts (adversarial task)

Creative features:
  - Urgency decay: P1s triaged late lose reward
  - Cascading context: hints that alerts are related
  - Persona: startup vs enterprise changes expected severity
  - Noise injection: garbled log entries test robustness
  - Adversarial: false positives disguised as real incidents
"""
from typing import Optional
from .models import Observation, Action, Reward, Alert
from .dataset import TASK_CONFIGS, ALL_ALERT_MAP, get_noisy_alert, CASCADING_CONTEXT
from .grader import grade


class IncidentResponseEnv:
    """
    OpenEnv interface:
      reset(task_id) -> Observation
      step(action)   -> (Observation | None, Reward, done: bool, info: dict)
      state()        -> dict
    """

    def __init__(self):
        self._task_id: Optional[str] = None
        self._alert_ids: list = []
        self._grade_fields: list = []
        self._persona: Optional[str] = None
        self._noise_level: float = 0.0
        self._step_idx: int = 0
        self._episode_rewards: list = []
        self._done: bool = False
        self._alert_cache: dict = {}

    def reset(self, task_id: str = "easy") -> Observation:
        cfg = TASK_CONFIGS[task_id]
        self._task_id = task_id
        self._alert_ids = cfg["alert_ids"][:]
        self._grade_fields = cfg["grade_fields"][:]
        self._persona = cfg.get("persona")
        self._noise_level = cfg.get("noise_level", 0.0)
        self._step_idx = 0
        self._episode_rewards = []
        self._done = False

        self._alert_cache = {}
        for aid in self._alert_ids:
            raw = ALL_ALERT_MAP[aid]
            self._alert_cache[aid] = (
                get_noisy_alert(raw, self._noise_level)
                if self._noise_level > 0 else raw
            )

        return self._make_obs()

    def step(self, action: Action) -> tuple:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        alert_id = self._alert_ids[self._step_idx]
        result = grade(
            alert_id=alert_id,
            action=action,
            grade_fields=self._grade_fields,
            step_idx=self._step_idx,
            persona=self._persona,
        )
        reward = Reward(value=result["reward"], breakdown=result["breakdown"])
        self._episode_rewards.append(reward.value)
        self._step_idx += 1

        done = self._step_idx >= len(self._alert_ids)
        self._done = done

        next_obs = None if done else self._make_obs()
        info = {
            "alert_id": alert_id,
            "step": self._step_idx,
            "persona": self._persona,
            "noise_level": self._noise_level,
            "cumulative_reward": round(sum(self._episode_rewards), 4),
        }
        return next_obs, reward, done, info

    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "step": self._step_idx,
            "total_steps": len(self._alert_ids),
            "done": self._done,
            "persona": self._persona,
            "noise_level": self._noise_level,
            "episode_rewards": self._episode_rewards,
            "mean_reward": round(
                sum(self._episode_rewards) / len(self._episode_rewards), 4
            ) if self._episode_rewards else 0.0,
        }

    def _make_obs(self) -> Observation:
        alert_id = self._alert_ids[self._step_idx]
        raw = self._alert_cache[alert_id]

        return Observation(
            alert=Alert(
                id=raw["id"],
                title=raw["title"],
                source=raw["source"],
                body=raw["body"],
                timestamp=raw["timestamp"],
                is_false_positive=False,  # never reveal ground truth
                noise_level=raw.get("noise_level", 0.0),
            ),
            step=self._step_idx,
            total_steps=len(self._alert_ids),
            task_id=self._task_id,
            persona=self._persona,
            cascading_context=CASCADING_CONTEXT.get(alert_id),
        )

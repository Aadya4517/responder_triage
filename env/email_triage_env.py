"""
EmailTriageEnv — OpenEnv-compliant environment for email triage.

Creative features:
  - Urgency decay: urgent emails triaged late get reduced reward
  - Adversarial detection: phishing emails disguised as legit
  - Inbox persona: agent role shifts expected triage behavior
  - Noise injection: garbled/truncated emails test robustness
"""
from typing import Optional
from .models import Observation, Action, Reward, Email
from .dataset import TASK_CONFIGS, ALL_EMAIL_MAP, get_noisy_email
from .grader import grade


class EmailTriageEnv:
    """
    OpenEnv interface:
      reset(task_id) -> Observation
      step(action)   -> (Observation | None, Reward, done: bool, info: dict)
      state()        -> dict
    """

    def __init__(self):
        self._task_id: Optional[str] = None
        self._email_ids: list[str] = []
        self._grade_fields: list[str] = []
        self._persona: Optional[str] = None
        self._noise_level: float = 0.0
        self._step_idx: int = 0
        self._episode_rewards: list[float] = []
        self._done: bool = False
        # Cache of (possibly noisy) email dicts for this episode
        self._email_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def reset(self, task_id: str = "easy") -> Observation:
        cfg = TASK_CONFIGS[task_id]
        self._task_id = task_id
        self._email_ids = cfg["email_ids"][:]
        self._grade_fields = cfg["grade_fields"][:]
        self._persona = cfg.get("persona")
        self._noise_level = cfg.get("noise_level", 0.0)
        self._step_idx = 0
        self._episode_rewards = []
        self._done = False

        # Pre-build noisy email cache for this episode (deterministic)
        self._email_cache = {}
        for eid in self._email_ids:
            raw = ALL_EMAIL_MAP[eid]
            self._email_cache[eid] = (
                get_noisy_email(raw, self._noise_level)
                if self._noise_level > 0
                else raw
            )

        return self._make_obs()

    # ------------------------------------------------------------------
    def step(self, action: Action) -> tuple[Optional[Observation], Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        email_id = self._email_ids[self._step_idx]
        result = grade(
            email_id=email_id,
            action=action,
            grade_fields=self._grade_fields,
            step_idx=self._step_idx,
            persona=self._persona,
        )
        reward = Reward(value=result["reward"], breakdown=result["breakdown"])
        self._episode_rewards.append(reward.value)
        self._step_idx += 1

        done = self._step_idx >= len(self._email_ids)
        self._done = done

        next_obs = None if done else self._make_obs()
        info = {
            "email_id": email_id,
            "step": self._step_idx,
            "persona": self._persona,
            "noise_level": self._noise_level,
            "cumulative_reward": round(sum(self._episode_rewards), 4),
        }
        return next_obs, reward, done, info

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "step": self._step_idx,
            "total_steps": len(self._email_ids),
            "done": self._done,
            "persona": self._persona,
            "noise_level": self._noise_level,
            "episode_rewards": self._episode_rewards,
            "mean_reward": round(
                sum(self._episode_rewards) / len(self._episode_rewards), 4
            ) if self._episode_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    def _make_obs(self) -> Observation:
        email_id = self._email_ids[self._step_idx]
        raw = self._email_cache[email_id]
        deadline = raw.get("urgency_deadline")

        return Observation(
            email=Email(
                id=raw["id"],
                subject=raw["subject"],
                sender=raw["sender"],
                body=raw["body"],
                timestamp=raw["timestamp"],
                is_adversarial=False,  # never reveal ground truth to agent
                noise_level=raw.get("noise_level", 0.0),
            ),
            step=self._step_idx,
            total_steps=len(self._email_ids),
            task_id=self._task_id,
            persona=self._persona,
            urgency_deadline=deadline,
        )

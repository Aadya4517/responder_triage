"""
CLI entrypoint: `openenv validate`
Runs the full spec compliance check across all 3 tasks.
"""
import sys
from env import EmailTriageEnv
from env.models import Action


def validate():
    print("Running openenv validate...\n")
    env = EmailTriageEnv()
    all_passed = True

    for task_id in ("easy", "medium", "hard"):
        try:
            obs = env.reset(task_id)
            assert obs is not None, "reset() returned None"
            assert obs.task_id == task_id, "task_id mismatch"
            assert obs.step == 0, "initial step should be 0"

            done = False
            steps = 0
            all_rewards = []

            while not done:
                assert obs is not None, "observation is None mid-episode"
                assert obs.email is not None, "observation missing email"
                action = Action(
                    priority="high",
                    category="support",
                    reply="Thank you for reaching out. Our team is on it.",
                )
                next_obs, reward, done, info = env.step(action)
                assert 0.0 <= reward.value <= 1.0, f"reward out of range: {reward.value}"
                assert isinstance(done, bool), "done must be bool"
                assert "email_id" in info, "info missing email_id"
                all_rewards.append(reward.value)
                steps += 1
                if not done:
                    assert next_obs is not None, "next_obs is None before episode end"
                    obs = next_obs

            state = env.state()
            assert state["done"] is True, "state.done should be True after episode"
            assert 0.0 <= state["mean_reward"] <= 1.0, "mean_reward out of range"

            print(f"  [PASS] task={task_id} steps={steps} mean_reward={state['mean_reward']:.4f}")

        except Exception as exc:
            print(f"  [FAIL] task={task_id} error={exc}")
            all_passed = False

    print()
    if all_passed:
        print("openenv validate: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("openenv validate: SOME CHECKS FAILED")
        sys.exit(1)


def cli():
    if len(sys.argv) < 2 or sys.argv[1] != "validate":
        print("Usage: openenv validate")
        sys.exit(1)
    validate()


if __name__ == "__main__":
    cli()

"""
openenv validate — smoke-tests all tasks against the OpenEnv spec.
"""
from env import EmailTriageEnv
from env.models import Action
from env.dataset import TASK_CONFIGS


def validate():
    print("Running openenv validate...\n")
    env = EmailTriageEnv()
    all_passed = True

    for task_id in TASK_CONFIGS:
        try:
            obs = env.reset(task_id)
            assert obs is not None, "reset() returned None"
            assert obs.task_id == task_id

            done = False
            steps = 0
            while not done:
                action = Action(
                    priority="high",
                    category="support",
                    reply="Thank you for reaching out. Our team is investigating.",
                    is_phishing=False,
                )
                next_obs, reward, done, info = env.step(action)
                assert 0.0 <= reward.value <= 1.0, f"reward out of range: {reward.value}"
                steps += 1
                obs = next_obs

            state = env.state()
            assert state["done"] is True
            assert 0.0 <= state["mean_reward"] <= 1.0

            print(f"  [PASS] task={task_id:<20} steps={steps} mean_reward={state['mean_reward']:.4f}")

        except Exception as exc:
            print(f"  [FAIL] task={task_id:<20} error={exc}")
            all_passed = False

    print()
    if all_passed:
        print("openenv validate: ALL CHECKS PASSED")
    else:
        print("openenv validate: SOME CHECKS FAILED")
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)

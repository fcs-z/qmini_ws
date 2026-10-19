"""List Gymnasium environments registered by qmini_lab."""

import gymnasium as gym

import qmini_lab  # noqa: F401


if __name__ == "__main__":
    task_ids = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("Qmini-"))
    if not task_ids:
        raise SystemExit("No Qmini tasks were registered.")
    print("Registered Qmini tasks:")
    for task_id in task_ids:
        print(f"  {task_id}")


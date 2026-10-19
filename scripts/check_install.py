"""Check the Python and package versions required by this workspace."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys


REQUIRED_DISTRIBUTIONS = ("isaacsim", "isaaclab", "rsl-rl-lib", "qmini-lab")


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    failed = sys.version_info[:2] != (3, 12)
    if failed:
        print("ERROR: Isaac Sim 6.0 / Isaac Lab 3.0 requires Python 3.12.")

    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            print(f"{distribution}: {metadata.version(distribution)}")
        except metadata.PackageNotFoundError:
            print(f"MISSING: {distribution}")
            failed = True

    if not failed:
        import gymnasium as gym
        import qmini_lab  # noqa: F401

        tasks = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("Qmini-"))
        print(f"Registered tasks: {len(tasks)}")
        for task in tasks:
            print(f"  {task}")
        failed = not tasks
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())


"""Qmini wrapper for Isaac Lab's version-matched RSL-RL player/exporter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _isaaclab_runner import run  # noqa: E402


if __name__ == "__main__":
    run("scripts/reinforcement_learning/rsl_rl/play.py")


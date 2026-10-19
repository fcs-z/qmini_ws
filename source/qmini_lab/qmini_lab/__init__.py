"""Qmini external extension for Isaac Lab.

Importing this package registers the Gymnasium environments. The package is
kept free of direct ``omni.*`` imports so registration can occur before the
Isaac Sim application is launched.
"""

from __future__ import annotations

import sys

from . import tasks as tasks

__all__ = ["register_tasks", "tasks"]


def register_tasks() -> list[str]:
    """Register tasks and return unconsumed CLI arguments.

    The return value makes this function usable as Isaac Lab's
    ``--external_callback qmini_lab.register_tasks`` callback.
    """

    return sys.argv[1:]


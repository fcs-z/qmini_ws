"""Run an upstream Isaac Lab script after registering qmini_lab tasks."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    if value := os.environ.get("ISAACLAB_ROOT"):
        candidates.append(Path(value).expanduser())

    project_root = Path(__file__).resolve().parents[1]
    candidates.extend((project_root.parent / "IsaacLab", Path.home() / "IsaacLab"))

    try:
        import isaaclab

        candidates.extend(Path(isaaclab.__file__).resolve().parents)
    except ImportError:
        pass

    unique: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in unique:
            unique.append(candidate)
    return unique


def run(relative_script: str) -> None:
    """Register this extension and execute a script from the Isaac Lab checkout."""

    try:
        import qmini_lab  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "qmini_lab is not installed. Run: python -m pip install -e source/qmini_lab"
        ) from exc

    script_path: Path | None = None
    for root in _candidate_roots():
        candidate = root / relative_script
        if candidate.is_file():
            script_path = candidate
            break

    if script_path is None:
        searched = "\n  - ".join(str(path / relative_script) for path in _candidate_roots())
        raise SystemExit(
            "Unable to locate the Isaac Lab checkout. Export ISAACLAB_ROOT first.\n"
            f"Searched:\n  - {searched}"
        )

    sys.path.insert(0, str(script_path.parent))
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")

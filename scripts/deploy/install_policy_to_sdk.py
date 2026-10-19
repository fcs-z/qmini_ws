"""Validate and copy an ONNX policy/config into RoboTamerSdk4Qmini/bin."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from clip_onnx_actions import clip_policy
from validate_onnx import validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path, help="Exported policy.onnx")
    parser.add_argument("sdk", type=Path, help="Path to RoboTamerSdk4Qmini")
    args = parser.parse_args()

    policy = args.policy.resolve()
    sdk_bin = args.sdk.resolve() / "bin"
    if not sdk_bin.is_dir():
        raise SystemExit(f"SDK bin directory does not exist: {sdk_bin}")
    deployment_policy = sdk_bin / "policy.onnx"
    clip_policy(policy, deployment_policy)
    validate(deployment_policy)
    shutil.copy2(Path(__file__).with_name("qmini_sdk_config.yaml"), sdk_bin / "config.yaml")
    print(f"Installed policy and config into {sdk_bin}")

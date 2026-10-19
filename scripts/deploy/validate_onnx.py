"""Validate an exported policy against the Qmini SDK tensor contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def validate(policy_path: Path) -> None:
    contract_path = Path(__file__).with_name("policy_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The deployment policy must have exactly one input and one output.")

    policy_input = session.get_inputs()[0]
    policy_output = session.get_outputs()[0]
    expected_input = contract["input_dimension"]
    expected_output = contract["output_dimension"]
    if policy_input.shape[-1] not in (expected_input, str(expected_input)):
        raise ValueError(f"Expected input width {expected_input}; received {policy_input.shape}.")
    if policy_output.shape[-1] not in (expected_output, str(expected_output)):
        raise ValueError(f"Expected output width {expected_output}; received {policy_output.shape}.")

    rng = np.random.default_rng(1)
    sample = rng.uniform(-3.0, 3.0, size=(32, expected_input)).astype(np.float32)
    output = session.run([policy_output.name], {policy_input.name: sample})[0]
    if output.shape != (32, expected_output) or not np.isfinite(output).all():
        raise ValueError(f"Invalid inference output: shape={output.shape}, finite={np.isfinite(output).all()}.")
    if np.max(np.abs(output)) > 1.00001:
        raise ValueError(
            "Policy output is not bounded to [-1, 1]. Run clip_onnx_actions.py before real-robot deployment."
        )
    print(f"OK: {policy_path}")
    print(f"  input:  {policy_input.name} {policy_input.shape}")
    print(f"  output: {policy_output.name} {policy_output.shape}")
    print(f"  sampled output range: [{output.min():.6f}, {output.max():.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path, help="Path to policy.onnx")
    args = parser.parse_args()
    validate(args.policy.resolve())

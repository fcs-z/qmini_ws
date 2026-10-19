"""Append an ONNX Clip node so SDK actions stay in the trained [-1, 1] range."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def clip_policy(source: Path, destination: Path) -> None:
    """Create a deployment model whose sole output is clipped to [-1, 1]."""

    model = onnx.load(str(source))
    if len(model.graph.output) != 1:
        raise ValueError("The Qmini policy must have exactly one graph output.")

    original_output = model.graph.output[0].name
    clipped_output = "normalized_action"
    if original_output == clipped_output:
        clipped_output = "normalized_action_clipped"
    min_name = "qmini_action_min"
    max_name = "qmini_action_max"
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=min_name),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=max_name),
        ]
    )
    model.graph.node.append(
        helper.make_node(
            "Clip",
            inputs=[original_output, min_name, max_name],
            outputs=[clipped_output],
            name="QminiDeploymentActionClip",
        )
    )
    model.graph.output[0].name = clipped_output
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="RSL-RL exported policy.onnx")
    parser.add_argument("destination", type=Path, help="Bounded deployment policy.onnx")
    args = parser.parse_args()
    clip_policy(args.source.resolve(), args.destination.resolve())
    print(f"Wrote bounded deployment policy: {args.destination.resolve()}")

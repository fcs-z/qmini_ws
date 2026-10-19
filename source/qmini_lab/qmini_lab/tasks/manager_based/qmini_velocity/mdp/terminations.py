"""Termination terms matching the Qmini Direct environment."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg


def base_too_low(env, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return height < minimum_height


def base_tilted(env, limit_angle: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    robot = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w.torch)
    return (torch.abs(roll) > limit_angle) | (torch.abs(pitch) > limit_angle)

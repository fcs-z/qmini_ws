"""Reward terms matching the Qmini Direct environment."""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg


def _commands(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    command = command[:, (0, 2)]
    return command * (command.abs() >= 0.15).to(command.dtype)


def _moving(env, command_name: str) -> torch.Tensor:
    import torch

    return (torch.linalg.norm(_commands(env, command_name), dim=-1) >= 0.15).float()


def track_lin_vel_x_exp(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    robot = env.scene[asset_cfg.name]
    error = torch.square(_commands(env, command_name)[:, 0] - robot.data.root_lin_vel_b.torch[:, 0])
    return torch.exp(-error / 0.25)


def track_yaw_rate_exp(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    robot = env.scene[asset_cfg.name]
    error = torch.square(_commands(env, command_name)[:, 1] - robot.data.root_ang_vel_b.torch[:, 2])
    return torch.exp(-error / 0.25)


def base_height_exp(env, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-70.0 * torch.square(height - target_height))


def upright_exp(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    robot = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w.torch)
    return torch.exp(-5.0 * (torch.square(roll) + torch.square(pitch)))


def lateral_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    return torch.square(env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, 1])


def vertical_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    return torch.square(env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    angular_velocity = env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, :2]
    return torch.sum(torch.square(angular_velocity), dim=-1)


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    reference = robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - reference), dim=-1)


def joint_velocity_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    joint_vel = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=-1)


def joint_torque_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    torque = env.scene[asset_cfg.name].data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torque), dim=-1)


def action_rate_l2(env, action_name: str) -> torch.Tensor:
    import torch

    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.raw_actions - action_term.previous_actions), dim=-1)


def gait_contact_match(
    env, action_name: str, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history.torch
    foot_force = torch.max(
        torch.linalg.norm(force_history[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1
    ).values
    foot_contact = foot_force > 10.0
    expected_support = env.action_manager.get_term(action_name).phase < math.pi
    return (foot_contact == expected_support).float().mean(dim=-1) * _moving(env, command_name)


def foot_clearance_l2(
    env,
    action_name: str,
    command_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    import torch

    robot = env.scene[asset_cfg.name]
    height = robot.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    height = height - env.scene.env_origins[:, 2].unsqueeze(-1)
    swing_mask = (env.action_manager.get_term(action_name).phase >= math.pi).float()
    return (
        torch.sum(torch.square((height - target_height) * swing_mask), dim=-1)
        * _moving(env, command_name)
    )


def foot_slip(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history.torch
    foot_force = torch.max(
        torch.linalg.norm(force_history[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1
    ).values
    foot_contact = foot_force > 10.0
    foot_velocity = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[
        :, asset_cfg.body_ids, :2
    ]
    return torch.sum(
        torch.linalg.norm(foot_velocity, dim=-1) * foot_contact.float(), dim=-1
    )


def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    import torch

    sensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.net_forces_w_history.torch
    force = torch.max(
        torch.linalg.norm(force_history[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1
    ).values
    return torch.sum((force > threshold).float(), dim=-1)

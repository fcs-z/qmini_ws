"""Observation terms for manager-based Qmini locomotion."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg


def qmini_policy_frame(
    env,
    action_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    enable_noise: bool = True,
    roll_pitch_noise: float = 0.15,
    angular_velocity_noise: float = 0.3,
    joint_position_noise: float = 0.1,
    joint_velocity_noise: float = 1.2,
) -> torch.Tensor:
    """Return one 43-value frame in the RoboTamer SDK order."""

    import torch
    from isaaclab.utils.math import euler_xyz_from_quat

    robot = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    velocity_command = env.command_manager.get_command(command_name)
    commands = velocity_command[:, (0, 2)]
    commands = commands * (commands.abs() >= 0.15).to(commands.dtype)

    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w.torch)
    roll_pitch = torch.stack((roll, pitch), dim=-1)
    angular_velocity = robot.data.root_ang_vel_b.torch
    joint_pos = robot.data.joint_pos.torch[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]

    if enable_noise:
        roll_pitch = roll_pitch + _uniform_noise(roll_pitch, roll_pitch_noise)
        angular_velocity = angular_velocity + _uniform_noise(
            angular_velocity, angular_velocity_noise
        )
        joint_pos = joint_pos + _uniform_noise(joint_pos, joint_position_noise)
        joint_vel = joint_vel + _uniform_noise(joint_vel, joint_velocity_noise)

    reference = robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    moving = (torch.linalg.norm(commands, dim=-1, keepdim=True) >= 0.15).float()
    phase_features = torch.cat(
        (torch.sin(action_term.phase), torch.cos(action_term.phase)), dim=-1
    ) * moving
    frequency_features = (action_term.frequency * 0.3 - 1.0) * moving

    observation = torch.cat(
        (
            commands,
            roll_pitch,
            angular_velocity * 0.5,
            joint_pos - reference,
            joint_vel * 0.1,
            action_term.joint_targets - joint_pos,
            phase_features,
            frequency_features,
        ),
        dim=-1,
    )
    if observation.shape[-1] != 43:
        raise RuntimeError(
            f"Qmini observation contract changed: expected 43, got {observation.shape[-1]}."
        )
    return observation


def _uniform_noise(tensor: torch.Tensor, magnitude: float) -> torch.Tensor:
    import torch

    return magnitude * (2.0 * torch.rand_like(tensor) - 1.0)

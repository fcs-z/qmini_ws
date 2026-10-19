"""Direct Isaac Lab environment for Qmini velocity locomotion.

The actor interface intentionally matches RoboTamerSdk4Qmini:

* 12 actions: two gait frequencies followed by ten joint increments.
* 43 values per observation frame, stacked oldest-to-newest for three frames.
* 10-joint order fixed by :data:`qmini_lab.assets.QMINI_JOINT_NAMES`.
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import warp as wp
from pxr import Sdf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.utils import find_first_matching_prim, get_current_stage
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .qmini_velocity_env_cfg import QminiVelocityFlatEnvCfg, QminiVelocityRoughEnvCfg


class QminiVelocityEnv(DirectRLEnv):
    """Velocity-command locomotion task with an SDK-compatible actor contract."""

    cfg: QminiVelocityFlatEnvCfg | QminiVelocityRoughEnvCfg

    def __init__(
        self,
        cfg: QminiVelocityFlatEnvCfg | QminiVelocityRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, resolved_joint_names = self._robot.find_joints(
            list(self.cfg.joint_names), preserve_order=True
        )
        if tuple(resolved_joint_names) != tuple(self.cfg.joint_names):
            raise RuntimeError(
                "Qmini joint order does not match the deployment contract. "
                f"Expected {self.cfg.joint_names}, received {resolved_joint_names}."
            )

        self._foot_body_ids, _ = self._robot.find_bodies(
            ["ankle_pitch_l", "ankle_pitch_r"], preserve_order=True
        )
        self._base_sensor_ids, _ = self._contact_sensor.find_sensors("base_link")
        self._foot_sensor_ids, _ = self._contact_sensor.find_sensors(
            ["ankle_pitch_l", "ankle_pitch_r"], preserve_order=True
        )
        self._undesired_sensor_ids, _ = self._contact_sensor.find_sensors(".*(hip|knee).*")

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 2, device=self.device)
        self._static_flag = torch.zeros(self.num_envs, 1, device=self.device)
        self._phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._frequency = torch.full((self.num_envs, 2), self.cfg.frequency_range[0], device=self.device)

        self._reference_joint_pos = torch.tensor(
            self.cfg.reference_joint_pos, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._joint_targets = self._robot.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._previous_joint_targets = self._joint_targets.clone()
        self._joint_pos_limits = self._robot.data.soft_joint_pos_limits.torch[:, self._joint_ids]

        self._obs_history = torch.zeros(
            self.num_envs,
            self.cfg.observation_history_length,
            43,
            dtype=torch.float32,
            device=self.device,
        )
        self._history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_resampling_steps = max(1, round(self.cfg.command_resampling_time / self.step_dt))

        reward_names = (
            "track_lin_vel",
            "track_yaw_rate",
            "base_height",
            "upright",
            "lateral_vel",
            "vertical_vel",
            "angular_vel",
            "joint_deviation",
            "joint_velocity",
            "joint_torque",
            "action_rate",
            "gait_contact",
            "foot_clearance",
            "foot_slip",
            "undesired_contact",
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device) for name in reward_names
        }

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._activate_articulation_contact_reporters()

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _activate_articulation_contact_reporters(self) -> None:
        """Attach contact reporting to the actual bodies targeted by USD joints.

        The Isaac Sim 6 URDF importer nests Qmini's rigid links.  Isaac Lab's
        generic file-spawner activation stops descending after the first rigid
        body, leaving only ``base_link`` visible to ``ContactSensor``.  Joint
        body relationships are the authoritative articulation body list and
        remain valid across importer hierarchy changes.
        """

        stage = get_current_stage()
        robot_prim = find_first_matching_prim(self.cfg.robot.prim_path, stage=stage)
        if robot_prim is None:
            raise RuntimeError(f"Unable to resolve Qmini prim: {self.cfg.robot.prim_path}")

        body_prims: dict[str, Usd.Prim] = {}
        for prim in Usd.PrimRange(robot_prim):
            joint = UsdPhysics.Joint(prim)
            if not joint:
                continue
            for relation in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                for target_path in relation.GetTargets():
                    body_prim = stage.GetPrimAtPath(target_path)
                    if body_prim.IsValid() and body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        body_prims[body_prim.GetPath().pathString] = body_prim

        expected_names = {"base_link", *self.cfg.joint_names}
        resolved_names = {prim.GetName() for prim in body_prims.values()}
        missing_names = expected_names - resolved_names
        if missing_names:
            resolved_paths = "\n  - ".join(sorted(body_prims)) or "<none>"
            raise RuntimeError(
                "Unable to resolve all Qmini articulation bodies from USD joint relationships. "
                f"Missing: {sorted(missing_names)}\nResolved paths:\n  - {resolved_paths}"
            )

        for body_prim in body_prims.values():
            applied_schemas = body_prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied_schemas:
                body_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            if "PhysxContactReportAPI" not in applied_schemas:
                body_prim.AddAppliedSchema("PhysxContactReportAPI")
            body_prim.CreateAttribute(
                "physxContactReport:threshold", Sdf.ValueTypeNames.Float
            ).Set(0.0)
        print(
            f"[INFO]: Enabled Qmini contact reporting on {len(body_prims)} bodies: "
            f"{sorted(resolved_names)}"
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._previous_joint_targets.copy_(self._joint_targets)

        resample_ids = torch.nonzero(
            (self.episode_length_buf > 0)
            & (self.episode_length_buf % self._command_resampling_steps == 0),
            as_tuple=False,
        ).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        frequency_alpha = 0.5 * (self._actions[:, :2] + 1.0)
        self._frequency = self.cfg.frequency_range[0] + frequency_alpha * (
            self.cfg.frequency_range[1] - self.cfg.frequency_range[0]
        )
        self._phase = torch.remainder(
            self._phase + (2.0 * math.pi * self._frequency * self.step_dt), 2.0 * math.pi
        )

        joint_increment = self._actions[:, 2:] * self.cfg.joint_increment_rate
        self._joint_targets += joint_increment * self.step_dt
        self._joint_targets = torch.clamp(
            self._joint_targets,
            min=self._joint_pos_limits[..., 0],
            max=self._joint_pos_limits[..., 1],
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(target=self._joint_targets, joint_ids=self._joint_ids)

    def _single_frame_observation(self) -> torch.Tensor:
        root_quat = self._robot.data.root_quat_w.torch
        roll, pitch, _ = euler_xyz_from_quat(root_quat)
        roll_pitch = torch.stack((roll, pitch), dim=-1)
        angular_velocity = self._robot.data.root_ang_vel_b.torch
        joint_pos = self._robot.data.joint_pos.torch[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._joint_ids]

        if self.cfg.observation_noise:
            roll_pitch = roll_pitch + self._uniform_noise(roll_pitch, self.cfg.roll_pitch_noise)
            angular_velocity = angular_velocity + self._uniform_noise(
                angular_velocity, self.cfg.angular_velocity_noise
            )
            joint_pos = joint_pos + self._uniform_noise(joint_pos, self.cfg.joint_position_noise)
            joint_vel = joint_vel + self._uniform_noise(joint_vel, self.cfg.joint_velocity_noise)

        joint_pos_error = self._joint_targets - joint_pos
        phase_features = torch.cat((torch.sin(self._phase), torch.cos(self._phase)), dim=-1)
        phase_features *= self._static_flag
        frequency_features = (self._frequency * 0.3 - 1.0) * self._static_flag

        observation = torch.cat(
            (
                self._commands,
                roll_pitch,
                angular_velocity * 0.5,
                joint_pos - self._reference_joint_pos,
                joint_vel * 0.1,
                joint_pos_error,
                phase_features,
                frequency_features,
            ),
            dim=-1,
        )
        if observation.shape[-1] != 43:
            raise RuntimeError(f"Qmini observation contract changed: expected 43, got {observation.shape[-1]}.")
        return observation.clamp(-3.0, 3.0)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        single_frame = self._single_frame_observation()
        self._obs_history[:, :-1] = self._obs_history[:, 1:].clone()
        self._obs_history[:, -1] = single_frame

        if torch.any(self._history_reset):
            fresh_history = single_frame.unsqueeze(1).expand(-1, self.cfg.observation_history_length, -1)
            self._obs_history[self._history_reset] = fresh_history[self._history_reset]
            self._history_reset[:] = False

        return {"policy": self._obs_history.flatten(start_dim=1)}

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel_b = self._robot.data.root_lin_vel_b.torch
        root_ang_vel_b = self._robot.data.root_ang_vel_b.torch
        root_pos_w = self._robot.data.root_pos_w.torch
        root_quat_w = self._robot.data.root_quat_w.torch
        roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
        roll_pitch = torch.stack((roll, pitch), dim=-1)
        base_height = root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

        lin_error = torch.square(self._commands[:, 0] - root_lin_vel_b[:, 0])
        yaw_error = torch.square(self._commands[:, 1] - root_ang_vel_b[:, 2])
        track_lin_vel = torch.exp(-lin_error / 0.25) * self.cfg.lin_vel_reward_scale
        track_yaw_rate = torch.exp(-yaw_error / 0.25) * self.cfg.yaw_rate_reward_scale
        base_height_reward = torch.exp(-70.0 * torch.square(base_height - self.cfg.target_base_height))
        base_height_reward *= self.cfg.base_height_reward_scale
        upright = torch.exp(-5.0 * torch.sum(torch.square(roll_pitch), dim=-1)) * self.cfg.upright_reward_scale

        joint_pos = self._robot.data.joint_pos.torch[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel.torch[:, self._joint_ids]
        joint_torque = self._robot.data.applied_torque.torch[:, self._joint_ids]
        lateral_vel = torch.square(root_lin_vel_b[:, 1]) * self.cfg.lateral_vel_penalty_scale
        vertical_vel = torch.square(root_lin_vel_b[:, 2]) * self.cfg.vertical_vel_penalty_scale
        angular_vel = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=-1) * self.cfg.angular_vel_penalty_scale
        joint_deviation = torch.sum(
            torch.square(joint_pos - self._reference_joint_pos), dim=-1
        ) * self.cfg.joint_deviation_penalty_scale
        joint_velocity = torch.sum(torch.square(joint_vel), dim=-1) * self.cfg.joint_velocity_penalty_scale
        joint_torque_penalty = torch.sum(torch.square(joint_torque), dim=-1) * self.cfg.joint_torque_penalty_scale
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)
        action_rate *= self.cfg.action_rate_penalty_scale

        force_history = self._contact_sensor.data.net_forces_w_history.torch
        foot_force = torch.max(
            torch.linalg.norm(force_history[:, :, self._foot_sensor_ids, :], dim=-1), dim=1
        ).values
        foot_contact = foot_force > 10.0
        expected_support = self._phase < math.pi
        moving = self._static_flag.squeeze(-1)
        gait_contact = (foot_contact == expected_support).float().mean(dim=-1) * moving
        gait_contact *= self.cfg.gait_contact_reward_scale

        foot_height = (
            self._robot.data.body_pos_w.torch[:, self._foot_body_ids, 2]
            - self._terrain.env_origins[:, 2].unsqueeze(-1)
        )
        swing_mask = (~expected_support).float()
        foot_clearance = torch.sum(torch.square((foot_height - 0.05) * swing_mask), dim=-1)
        foot_clearance *= self.cfg.foot_clearance_penalty_scale * moving
        foot_xy_vel = self._robot.data.body_lin_vel_w.torch[:, self._foot_body_ids, :2]
        foot_slip = torch.sum(torch.linalg.norm(foot_xy_vel, dim=-1) * foot_contact.float(), dim=-1)
        foot_slip *= self.cfg.foot_slip_penalty_scale

        undesired_force = torch.max(
            torch.linalg.norm(force_history[:, :, self._undesired_sensor_ids, :], dim=-1), dim=1
        ).values
        undesired_contact = torch.sum((undesired_force > 1.0).float(), dim=-1)
        undesired_contact *= self.cfg.undesired_contact_penalty_scale

        terms = {
            "track_lin_vel": track_lin_vel,
            "track_yaw_rate": track_yaw_rate,
            "base_height": base_height_reward,
            "upright": upright,
            "lateral_vel": lateral_vel,
            "vertical_vel": vertical_vel,
            "angular_vel": angular_vel,
            "joint_deviation": joint_deviation,
            "joint_velocity": joint_velocity,
            "joint_torque": joint_torque_penalty,
            "action_rate": action_rate,
            "gait_contact": gait_contact,
            "foot_clearance": foot_clearance,
            "foot_slip": foot_slip,
            "undesired_contact": undesired_contact,
        }
        rewards = {name: value * self.step_dt for name, value in terms.items()}
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.sum(torch.stack(tuple(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self._robot.data.root_pos_w.torch
        root_quat_w = self._robot.data.root_quat_w.torch
        roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
        base_height = root_pos_w[:, 2] - self._terrain.env_origins[:, 2]

        force_history = self._contact_sensor.data.net_forces_w_history.torch
        torso_force = torch.max(
            torch.linalg.norm(force_history[:, :, self._base_sensor_ids, :], dim=-1), dim=1
        ).values
        torso_contact = torch.any(torso_force > 1.0, dim=-1)
        undesired_force = torch.max(
            torch.linalg.norm(force_history[:, :, self._undesired_sensor_ids, :], dim=-1), dim=1
        ).values
        undesired_contact = torch.any(undesired_force > 1.0, dim=-1)

        too_low = base_height < self.cfg.termination_height
        too_tilted = (torch.abs(roll) > self.cfg.termination_tilt) | (
            torch.abs(pitch) > self.cfg.termination_tilt
        )
        self._termination_reasons = {
            "base_too_low": too_low,
            "base_tilted": too_tilted,
            "torso_contact": torso_contact,
            "hip_or_knee_contact": undesired_contact,
        }
        terminated = too_low | too_tilted | torso_contact | undesired_contact
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)

        if hasattr(self, "_episode_sums"):
            self.extras["log"] = {}
            for name, values in self._episode_sums.items():
                self.extras["log"][f"Episode_Reward/{name}"] = (
                    torch.mean(values[env_ids]) / self.max_episode_length_s
                ).item()
                values[env_ids] = 0.0
            if hasattr(self, "_termination_reasons"):
                for name, values in self._termination_reasons.items():
                    self.extras["log"][f"Episode_Termination/{name}"] = torch.count_nonzero(
                        values[env_ids]
                    ).item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel.torch[env_ids].clone()
        controlled_joint_pos = joint_pos[:, self._joint_ids]
        controlled_joint_pos += 0.03 * (2.0 * torch.rand_like(controlled_joint_pos) - 1.0)
        controlled_limits = self._joint_pos_limits[env_ids]
        controlled_joint_pos.clamp_(controlled_limits[..., 0], controlled_limits[..., 1])
        joint_pos[:, self._joint_ids] = controlled_joint_pos

        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_velocity = self._robot.data.default_root_vel.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        reset_rpy = 0.10 * (
            2.0 * torch.rand(len(env_ids), 3, device=self.device, dtype=torch.float32) - 1.0
        )
        root_pose[:, 3:7] = quat_from_euler_xyz(reset_rpy[:, 0], reset_rpy[:, 1], reset_rpy[:, 2])
        root_velocity += 0.05 * (2.0 * torch.rand_like(root_velocity) - 1.0)

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._reference_joint_pos[env_ids]
        self._previous_joint_targets[env_ids] = self._reference_joint_pos[env_ids]
        self._robot.set_joint_position_target_index(
            target=self._joint_targets[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )
        self._frequency[env_ids] = self.cfg.frequency_range[0]
        if self.cfg.randomize_phase_on_reset:
            self._phase[env_ids] = 2.0 * math.pi * torch.rand(len(env_ids), 2, device=self.device)
        else:
            self._phase[env_ids] = 0.0
        self._resample_commands(env_ids)
        self._history_reset[env_ids] = True

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        self._commands[env_ids, 0] = torch.empty(count, device=self.device).uniform_(
            self.cfg.command_vx_range[0], self.cfg.command_vx_range[1]
        )
        self._commands[env_ids, 1] = torch.empty(count, device=self.device).uniform_(
            self.cfg.command_yaw_range[0], self.cfg.command_yaw_range[1]
        )
        self._commands[env_ids] *= (
            torch.abs(self._commands[env_ids]) >= self.cfg.command_deadband
        ).float()
        stand_mask = torch.rand(count, device=self.device) < self.cfg.stand_command_fraction
        self._commands[env_ids[stand_mask]] = 0.0
        self._static_flag[env_ids] = (
            torch.linalg.norm(self._commands[env_ids], dim=-1, keepdim=True) >= self.cfg.command_deadband
        ).float()

    @staticmethod
    def _uniform_noise(tensor: torch.Tensor, magnitude: float) -> torch.Tensor:
        return magnitude * (2.0 * torch.rand_like(tensor) - 1.0)

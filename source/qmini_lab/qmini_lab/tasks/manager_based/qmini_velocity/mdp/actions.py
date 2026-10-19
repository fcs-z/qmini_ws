"""SDK-compatible manager action term for Qmini."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    import torch
    from isaaclab.assets import Articulation


class QminiGaitAction(ActionTerm):
    """Map 12 policy actions to gait frequency and integrated joint targets."""

    cfg: QminiGaitActionCfg
    _asset: Articulation

    def __init__(self, cfg: QminiGaitActionCfg, env):
        import torch

        super().__init__(cfg, env)

        self._joint_ids, resolved_names = self._asset.find_joints(
            list(cfg.joint_names), preserve_order=True
        )
        if tuple(resolved_names) != tuple(cfg.joint_names):
            raise RuntimeError(
                "Qmini joint order does not match the deployment contract. "
                f"Expected {cfg.joint_names}, received {resolved_names}."
            )

        self._raw_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._frequency = torch.full(
            (self.num_envs, 2), cfg.frequency_range[0], device=self.device
        )
        self._reference_joint_pos = torch.tensor(
            cfg.reference_joint_pos, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._joint_targets = self._reference_joint_pos.clone()
        self._joint_pos_limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def previous_actions(self) -> torch.Tensor:
        return self._previous_actions

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    @property
    def frequency(self) -> torch.Tensor:
        return self._frequency

    @property
    def joint_targets(self) -> torch.Tensor:
        return self._joint_targets

    @property
    def joint_ids(self) -> list[int] | slice:
        return self._joint_ids

    def process_actions(self, actions: torch.Tensor) -> None:
        import torch

        self._previous_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        self._processed_actions.copy_(self._raw_actions)

        frequency_alpha = 0.5 * (self._raw_actions[:, :2] + 1.0)
        self._frequency = self.cfg.frequency_range[0] + frequency_alpha * (
            self.cfg.frequency_range[1] - self.cfg.frequency_range[0]
        )
        self._phase = torch.remainder(
            self._phase + 2.0 * math.pi * self._frequency * self._env.step_dt,
            2.0 * math.pi,
        )

        self._joint_targets += (
            self._raw_actions[:, 2:] * self.cfg.joint_increment_rate * self._env.step_dt
        )
        self._joint_targets.clamp_(
            min=self._joint_pos_limits[..., 0], max=self._joint_pos_limits[..., 1]
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._joint_targets, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        import torch

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        count = len(env_ids)

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._reference_joint_pos[env_ids]
        self._frequency[env_ids] = self.cfg.frequency_range[0]
        if self.cfg.randomize_phase_on_reset:
            self._phase[env_ids] = 2.0 * math.pi * torch.rand(count, 2, device=self.device)
        else:
            self._phase[env_ids] = 0.0
        self._asset.set_joint_position_target_index(
            target=self._joint_targets[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )


@configclass
class QminiGaitActionCfg(ActionTermCfg):
    """Configuration for :class:`QminiGaitAction`."""

    class_type: type[ActionTerm] = QminiGaitAction
    asset_name: str = "robot"
    joint_names: tuple[str, ...] = ()
    reference_joint_pos: tuple[float, ...] = ()
    frequency_range: tuple[float, float] = (0.5, 3.5)
    joint_increment_rate: float = 15.0
    randomize_phase_on_reset: bool = True

"""Velocity command term with the Qmini deployment deadband."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass


class QminiVelocityCommand(UniformVelocityCommand):
    """Uniform planar command with component-wise deadband removal."""

    cfg: QminiVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        commands = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = commands * (
            commands.abs() >= self.cfg.deadband
        ).float()


@configclass
class QminiVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`QminiVelocityCommand`."""

    class_type: type[UniformVelocityCommand] = QminiVelocityCommand
    deadband: float = 0.15

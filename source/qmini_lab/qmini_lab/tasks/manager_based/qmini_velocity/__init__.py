"""Gymnasium registrations for manager-based Qmini velocity locomotion."""

import gymnasium as gym

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
_RSL_RL_CFG = f"{__name__}.agents.rsl_rl_ppo_cfg"

gym.register(
    id="Qmini-Velocity-Flat-Manager-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityFlatManagerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_RSL_RL_CFG}:QminiFlatManagerPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Flat-Manager-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityFlatManagerPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_RSL_RL_CFG}:QminiFlatManagerPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Rough-Manager-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityRoughManagerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_RSL_RL_CFG}:QminiRoughManagerPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Rough-Manager-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityRoughManagerPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_RSL_RL_CFG}:QminiRoughManagerPPORunnerCfg",
    },
)

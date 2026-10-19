"""Gymnasium registrations for Qmini velocity locomotion."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = f"{__name__}.qmini_velocity_env:QminiVelocityEnv"

gym.register(
    id="Qmini-Velocity-Flat-Direct-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiFlatPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Flat-Direct-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiFlatPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Rough-Direct-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiRoughPPORunnerCfg",
    },
)

gym.register(
    id="Qmini-Velocity-Rough-Direct-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_velocity_env_cfg:QminiVelocityRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiRoughPPORunnerCfg",
    },
)

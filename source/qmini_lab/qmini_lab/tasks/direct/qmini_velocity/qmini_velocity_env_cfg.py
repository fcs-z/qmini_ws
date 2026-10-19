"""Configuration for Qmini velocity-tracking locomotion."""

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from qmini_lab.assets import QMINI_CFG, QMINI_DEFAULT_JOINT_POS, QMINI_JOINT_NAMES


@configclass
class QminiEventCfg:
    """Startup and interval domain randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 3.5),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    )


@configclass
class QminiVelocityFlatEnvCfg(DirectRLEnvCfg):
    # Environment and policy spaces. One frame has 43 observations and the
    # deployed policy consumes three frames: 43 * 3 = 129.
    episode_length_s = 10.0
    decimation = 15
    action_space = 12
    observation_space = 129
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    # DirectRLEnv constructs the articulation itself, so this must be an
    # absolute stage regex.  The {ENV_REGEX_NS} placeholder is only expanded
    # when an asset config is resolved by InteractiveScene.
    robot: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.001,
        track_air_time=True,
        force_threshold=1.0,
    )
    events: QminiEventCfg | None = QminiEventCfg()

    # Policy/deployment contract.
    joint_names = QMINI_JOINT_NAMES
    reference_joint_pos = tuple(QMINI_DEFAULT_JOINT_POS[name] for name in QMINI_JOINT_NAMES)
    observation_history_length = 3
    frequency_range = (0.5, 3.5)
    joint_increment_rate = 15.0  # rad/s at normalized action magnitude 1.

    # Commands.
    command_resampling_time = 5.0
    command_vx_range = (-0.3, 0.7)
    command_yaw_range = (-1.0, 1.0)
    command_deadband = 0.15
    stand_command_fraction = 0.025

    # Reset and observation randomization.
    randomize_phase_on_reset = True
    observation_noise = True
    roll_pitch_noise = 0.15
    angular_velocity_noise = 0.3
    joint_position_noise = 0.1
    joint_velocity_noise = 1.2

    # Reward parameters. All terms are multiplied by the environment step.
    target_base_height = 0.45
    lin_vel_reward_scale = 2.3
    yaw_rate_reward_scale = 2.5
    base_height_reward_scale = 1.0
    upright_reward_scale = 1.5
    lateral_vel_penalty_scale = -0.5
    vertical_vel_penalty_scale = -0.6
    angular_vel_penalty_scale = -0.1
    joint_deviation_penalty_scale = -0.02
    joint_velocity_penalty_scale = -0.0012
    joint_torque_penalty_scale = -1.0e-5
    action_rate_penalty_scale = -0.015
    gait_contact_reward_scale = 0.35
    foot_clearance_penalty_scale = -1.0
    foot_slip_penalty_scale = -0.08
    undesired_contact_penalty_scale = -1.0

    termination_height = 0.20
    termination_tilt = 0.70


@configclass
class QminiVelocityRoughEnvCfg(QminiVelocityFlatEnvCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class QminiVelocityFlatPlayEnvCfg(QminiVelocityFlatEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )
    events = None
    observation_noise = False
    randomize_phase_on_reset = False


@configclass
class QminiVelocityRoughPlayEnvCfg(QminiVelocityRoughEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )
    events = None
    observation_noise = False
    randomize_phase_on_reset = False

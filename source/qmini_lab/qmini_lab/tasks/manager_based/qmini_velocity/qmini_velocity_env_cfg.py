"""Manager-based Qmini velocity-tracking locomotion configuration."""

from __future__ import annotations

import copy

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as lab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from qmini_lab.assets import QMINI_CFG, QMINI_DEFAULT_JOINT_POS, QMINI_JOINT_NAMES

from . import mdp as qmini_mdp


_REFERENCE_JOINT_POS = tuple(QMINI_DEFAULT_JOINT_POS[name] for name in QMINI_JOINT_NAMES)
_FOOT_BODIES = ["ankle_pitch_l", "ankle_pitch_r"]


@configclass
class QminiContactArticulationCfg(ArticulationCfg):
    """Articulation config that authors contact schemas before scene cloning."""

    def _post_spawn(self, stage) -> None:
        # Environment configurations are imported before SimulationApp starts.
        # Import USD modules only inside this post-spawn hook, which runs after
        # Kit is initialized; importing pxr at module scope can crash Isaac Sim.
        from isaaclab.sim.utils import find_first_matching_prim
        from pxr import Sdf, Usd, UsdPhysics

        super()._post_spawn(stage)
        spawn_path = (
            self.spawn.spawn_path
            if self.spawn is not None and self.spawn.spawn_path is not None
            else self.prim_path
        )
        robot_prim = find_first_matching_prim(spawn_path, stage=stage)
        if robot_prim is None:
            raise RuntimeError(f"Unable to resolve spawned Qmini prim: {spawn_path}")

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

        expected_names = {"base_link", *QMINI_JOINT_NAMES}
        resolved_names = {prim.GetName() for prim in body_prims.values()}
        missing_names = expected_names - resolved_names
        if missing_names:
            resolved_paths = "\n  - ".join(sorted(body_prims)) or "<none>"
            raise RuntimeError(
                "Unable to resolve all Qmini articulation bodies before scene cloning. "
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
            f"[INFO]: Enabled Qmini contact reporting on {len(body_prims)} source bodies "
            "before Manager-Based scene cloning."
        )


def _manager_robot_cfg() -> QminiContactArticulationCfg:
    """Create an independent manager asset config without mutating the Direct task."""

    return QminiContactArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=copy.deepcopy(QMINI_CFG.spawn),
        init_state=copy.deepcopy(QMINI_CFG.init_state),
        collision_group=QMINI_CFG.collision_group,
        debug_vis=QMINI_CFG.debug_vis,
        soft_joint_pos_limit_factor=QMINI_CFG.soft_joint_pos_limit_factor,
        actuators=copy.deepcopy(QMINI_CFG.actuators),
    )


def _flat_terrain_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
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


def _rough_terrain_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=copy.deepcopy(ROUGH_TERRAINS_CFG),
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
class QminiFlatSceneCfg(InteractiveSceneCfg):
    """Qmini, contact sensing, terrain, and lighting."""

    terrain = _flat_terrain_cfg()
    robot: ArticulationCfg = _manager_robot_cfg()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.001,
        track_air_time=True,
        force_threshold=1.0,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class QminiRoughSceneCfg(QminiFlatSceneCfg):
    terrain = _rough_terrain_cfg()


@configclass
class QminiCommandsCfg:
    base_velocity = lab_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.025,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class QminiActionsCfg:
    qmini_action = qmini_mdp.QminiGaitActionCfg(
        asset_name="robot",
        joint_names=QMINI_JOINT_NAMES,
        reference_joint_pos=_REFERENCE_JOINT_POS,
        frequency_range=(0.5, 3.5),
        joint_increment_rate=15.0,
        randomize_phase_on_reset=True,
    )


@configclass
class QminiObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        qmini_frame = ObsTerm(
            func=qmini_mdp.qmini_policy_frame,
            params={
                "action_name": "qmini_action",
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(QMINI_JOINT_NAMES), preserve_order=True
                ),
                "enable_noise": True,
                "roll_pitch_noise": 0.15,
                "angular_velocity_noise": 0.3,
                "joint_position_noise": 0.1,
                "joint_velocity_noise": 1.2,
            },
            clip=(-3.0, 3.0),
            history_length=3,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.concatenate_terms = True
            # Noise is component-specific and is applied inside qmini_policy_frame.
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class QminiRewardsCfg:
    track_lin_vel = RewTerm(
        func=qmini_mdp.track_lin_vel_x_exp,
        weight=2.3,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    track_yaw_rate = RewTerm(
        func=qmini_mdp.track_yaw_rate_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewTerm(
        func=qmini_mdp.base_height_exp,
        weight=1.0,
        params={"target_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )
    upright = RewTerm(
        func=qmini_mdp.upright_exp,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lateral_vel = RewTerm(
        func=qmini_mdp.lateral_velocity_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    vertical_vel = RewTerm(
        func=qmini_mdp.vertical_velocity_l2,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    angular_vel = RewTerm(
        func=qmini_mdp.ang_vel_xy_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_deviation = RewTerm(
        func=qmini_mdp.joint_deviation_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(QMINI_JOINT_NAMES), preserve_order=True
            ),
        },
    )
    joint_velocity = RewTerm(
        func=qmini_mdp.joint_velocity_l2,
        weight=-0.0012,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(QMINI_JOINT_NAMES), preserve_order=True
            )
        },
    )
    joint_torque = RewTerm(
        func=qmini_mdp.joint_torque_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(QMINI_JOINT_NAMES), preserve_order=True
            )
        },
    )
    action_rate = RewTerm(
        func=qmini_mdp.action_rate_l2,
        weight=-0.015,
        params={"action_name": "qmini_action"},
    )
    gait_contact = RewTerm(
        func=qmini_mdp.gait_contact_match,
        weight=0.35,
        params={
            "action_name": "qmini_action",
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=_FOOT_BODIES, preserve_order=True
            ),
        },
    )
    foot_clearance = RewTerm(
        func=qmini_mdp.foot_clearance_l2,
        weight=-1.0,
        params={
            "action_name": "qmini_action",
            "command_name": "base_velocity",
            "target_height": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=_FOOT_BODIES, preserve_order=True
            ),
        },
    )
    foot_slip = RewTerm(
        func=qmini_mdp.foot_slip,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=_FOOT_BODIES, preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=_FOOT_BODIES, preserve_order=True
            ),
        },
    )
    undesired_contact = RewTerm(
        func=qmini_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*(hip|knee).*"
            ),
        },
    )


@configclass
class QminiTerminationsCfg:
    time_out = DoneTerm(func=lab_mdp.time_out, time_out=True)
    base_too_low = DoneTerm(
        func=qmini_mdp.base_too_low,
        params={"minimum_height": 0.20, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_tilted = DoneTerm(
        func=qmini_mdp.base_tilted,
        params={"limit_angle": 0.70, "asset_cfg": SceneEntityCfg("robot")},
    )
    torso_contact = DoneTerm(
        func=lab_mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
        },
    )
    hip_or_knee_contact = DoneTerm(
        func=lab_mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*(hip|knee).*"
            ),
        },
    )


@configclass
class QminiEventsCfg:
    physics_material = EventTerm(
        func=lab_mdp.randomize_rigid_body_material,
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
        func=lab_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    actuator_gains = EventTerm(
        func=lab_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=lab_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=lab_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.03, 0.03), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=lab_mdp.push_by_setting_velocity,
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
class QminiVelocityFlatManagerEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for flat terrain."""

    scene: QminiFlatSceneCfg = QminiFlatSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    observations: QminiObservationsCfg = QminiObservationsCfg()
    actions: QminiActionsCfg = QminiActionsCfg()
    commands: QminiCommandsCfg = QminiCommandsCfg()
    rewards: QminiRewardsCfg = QminiRewardsCfg()
    terminations: QminiTerminationsCfg = QminiTerminationsCfg()
    events: QminiEventsCfg = QminiEventsCfg()
    curriculum = None

    episode_length_s = 10.0
    decimation = 15
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


@configclass
class QminiVelocityRoughManagerEnvCfg(QminiVelocityFlatManagerEnvCfg):
    scene: QminiRoughSceneCfg = QminiRoughSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )


def _configure_play(cfg: QminiVelocityFlatManagerEnvCfg) -> None:
    cfg.scene.num_envs = 32
    cfg.actions.qmini_action.randomize_phase_on_reset = False
    cfg.observations.policy.qmini_frame.params["enable_noise"] = False

    # Keep deterministic reset events; ManagerBasedRLEnv relies on them to write
    # the default root and joint state back into the simulator.
    cfg.events.physics_material = None
    cfg.events.base_mass = None
    cfg.events.actuator_gains = None
    cfg.events.joint_friction = None
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.reset_joints.params["position_range"] = (0.0, 0.0)


@configclass
class QminiVelocityFlatManagerPlayEnvCfg(QminiVelocityFlatManagerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class QminiVelocityRoughManagerPlayEnvCfg(QminiVelocityRoughManagerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)

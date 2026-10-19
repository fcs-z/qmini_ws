"""Qmini articulation configuration for Isaac Sim 6.0 / Isaac Lab 3.0."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


QMINI_ASSET_DIR = Path(__file__).resolve().parent / "qmini"
QMINI_URDF_PATH = str(QMINI_ASSET_DIR / "urdf" / "qmini.urdf")

# This order is the policy/SDK contract. Do not reorder without retraining and
# updating RoboTamerSdk4Qmini's jointIndex2Sim mapping.
QMINI_JOINT_NAMES = (
    "hip_yaw_l",
    "hip_roll_l",
    "hip_pitch_l",
    "knee_pitch_l",
    "ankle_pitch_l",
    "hip_yaw_r",
    "hip_roll_r",
    "hip_pitch_r",
    "knee_pitch_r",
    "ankle_pitch_r",
)

QMINI_DEFAULT_JOINT_POS = {
    "hip_yaw_l": 0.4,
    "hip_roll_l": -0.1,
    "hip_pitch_l": -1.5,
    "knee_pitch_l": 1.0,
    "ankle_pitch_l": -1.3,
    "hip_yaw_r": -0.4,
    "hip_roll_r": 0.1,
    "hip_pitch_r": 1.5,
    "knee_pitch_r": -1.0,
    "ankle_pitch_r": 1.3,
}

_URDF_GAINS = sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
    stiffness={
        "hip_yaw_.*": 55.0,
        "hip_roll_.*": 105.0,
        "hip_pitch_.*": 75.0,
        "knee_pitch_.*": 45.0,
        "ankle_pitch_.*": 30.0,
    },
    damping={
        "hip_yaw_.*": 0.3,
        "hip_roll_.*": 2.5,
        "hip_pitch_.*": 0.3,
        "knee_pitch_.*": 0.5,
        "ankle_pitch_.*": 0.25,
    },
)

QMINI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=QMINI_URDF_PATH,
        # Qmini's imported rigid links are nested. Isaac Lab's generic contact
        # activation stops walking below the first rigid body and consequently
        # enables only base_link. The environment enables reporters explicitly
        # on every body referenced by the articulation joints instead.
        activate_contact_sensors=False,
        fix_base=False,
        merge_fixed_joints=True,
        # Isaac Sim 6's transformer emits a layered/instanceable USD.  For
        # this SolidWorks/STL URDF, the PhysX contact sensor then discovers
        # only the non-instanced base_link and cannot resolve the feet.  Keep
        # the imported articulation flat so every rigid link remains visible
        # to ContactSensor (base + ten actuated leg links).
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
        self_collision=True,
        collision_type="Convex Hull",
        robot_type="Humanoid",
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=_URDF_GAINS,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(0.0, 0.0, 0.0, 1.0),  # Isaac Lab 3.0 uses XYZW.
        joint_pos=QMINI_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.98,
    actuators={
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["hip_yaw_.*"], stiffness=55.0, damping=0.3, armature=0.0
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["hip_roll_.*"], stiffness=105.0, damping=2.5, armature=0.0
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["hip_pitch_.*"], stiffness=75.0, damping=0.3, armature=0.0
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["knee_pitch_.*"], stiffness=45.0, damping=0.5, armature=0.0
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*"], stiffness=30.0, damping=0.25, armature=0.0
        ),
    },
)

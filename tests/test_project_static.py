"""Static tests that can run without Isaac Sim or Isaac Lab installed."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "source" / "qmini_lab" / "qmini_lab" / "assets" / "qmini"
URDF = ASSET_ROOT / "urdf" / "qmini.urdf"
CONTRACT = ROOT / "scripts" / "deploy" / "policy_contract.json"
ENV_CFG = (
    ROOT
    / "source"
    / "qmini_lab"
    / "qmini_lab"
    / "tasks"
    / "direct"
    / "qmini_velocity"
    / "qmini_velocity_env_cfg.py"
)
ASSET_CFG = ROOT / "source" / "qmini_lab" / "qmini_lab" / "assets" / "qmini.py"
ENV_IMPL = ENV_CFG.with_name("qmini_velocity_env.py")
MANAGER_ROOT = (
    ROOT
    / "source"
    / "qmini_lab"
    / "qmini_lab"
    / "tasks"
    / "manager_based"
    / "qmini_velocity"
)
MANAGER_CFG = MANAGER_ROOT / "qmini_velocity_env_cfg.py"
MANAGER_REGISTRATION = MANAGER_ROOT / "__init__.py"
MANAGER_ACTIONS = MANAGER_ROOT / "mdp" / "actions.py"
TASKS_INIT = ROOT / "source" / "qmini_lab" / "qmini_lab" / "tasks" / "__init__.py"
DIRECT_REGISTRATION = MANAGER_ROOT.parents[1] / "direct" / "qmini_velocity" / "__init__.py"
RUNNER = ROOT / "scripts" / "_isaaclab_runner.py"


class ProjectContractTest(unittest.TestCase):
    def test_policy_dimensions(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["single_frame_dimension"] * contract["history_length"], 129)
        self.assertEqual(contract["input_dimension"], 129)
        self.assertEqual(contract["output_dimension"], 12)
        self.assertEqual(len(contract["joint_order"]), 10)
        self.assertEqual(len(contract["action_order"]), 12)

    def test_urdf_joint_order_and_limits(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        root = ET.parse(URDF).getroot()
        joints = [joint for joint in root.findall("joint") if joint.attrib.get("type") == "revolute"]
        self.assertEqual([joint.attrib["name"] for joint in joints], contract["joint_order"])
        for joint in joints:
            limit = joint.find("limit")
            self.assertIsNotNone(limit)
            self.assertLess(float(limit.attrib["lower"]), float(limit.attrib["upper"]))
            self.assertGreater(float(limit.attrib["effort"]), 0.0)

    def test_all_meshes_exist(self) -> None:
        root = ET.parse(URDF).getroot()
        for mesh in root.findall(".//mesh"):
            mesh_path = (URDF.parent / mesh.attrib["filename"]).resolve()
            self.assertTrue(mesh_path.is_file(), mesh_path)

    def test_torso_has_collision(self) -> None:
        root = ET.parse(URDF).getroot()
        base_link = next(link for link in root.findall("link") if link.attrib["name"] == "base_link")
        collision = base_link.find("collision")
        self.assertIsNotNone(collision)
        self.assertIsNotNone(collision.find("geometry/box"))
        self.assertIsNone(collision.find("geometry/mesh"))

    def test_direct_scene_prim_paths_are_global(self) -> None:
        source = ENV_CFG.read_text(encoding="utf-8")
        self.assertIn('prim_path="/World/envs/env_.*/Robot"', source)
        self.assertIn('prim_path="/World/envs/env_.*/Robot/.*"', source)
        self.assertNotIn('prim_path="{ENV_REGEX_NS}/Robot', source)

    def test_contact_reporting_is_enabled(self) -> None:
        asset_source = ASSET_CFG.read_text(encoding="utf-8")
        env_source = ENV_IMPL.read_text(encoding="utf-8")
        self.assertIn("activate_contact_sensors=False", asset_source)
        self.assertIn("run_asset_transformer=False", asset_source)
        self.assertIn("run_multi_physics_conversion=False", asset_source)
        self.assertIn("_activate_articulation_contact_reporters", env_source)
        self.assertIn('body_prim.AddAppliedSchema("PhysxContactReportAPI")', env_source)

    def test_manager_based_task_contract(self) -> None:
        registration = MANAGER_REGISTRATION.read_text(encoding="utf-8")
        cfg_source = MANAGER_CFG.read_text(encoding="utf-8")
        action_source = MANAGER_ACTIONS.read_text(encoding="utf-8")

        task_ids = (
            "Qmini-Velocity-Flat-Manager-v0",
            "Qmini-Velocity-Flat-Manager-Play-v0",
            "Qmini-Velocity-Rough-Manager-v0",
            "Qmini-Velocity-Rough-Manager-Play-v0",
        )
        for task_id in task_ids:
            self.assertIn(f'id="{task_id}"', registration)
        self.assertIn('"isaaclab.envs:ManagerBasedRLEnv"', registration)
        self.assertIn('prim_path="{ENV_REGEX_NS}/Robot"', cfg_source)
        self.assertIn("history_length=3", cfg_source)
        self.assertIn('body_prim.AddAppliedSchema("PhysxContactReportAPI")', cfg_source)
        self.assertNotIn("\nfrom pxr import", cfg_source)
        self.assertIn("        from pxr import Sdf, Usd, UsdPhysics", cfg_source)
        self.assertIn("def action_dim(self) -> int:", action_source)
        self.assertIn("return 12", action_source)

    def test_pre_app_registration_does_not_import_simulation_packages(self) -> None:
        tasks_source = TASKS_INIT.read_text(encoding="utf-8")
        runner_source = RUNNER.read_text(encoding="utf-8")
        direct_registration = DIRECT_REGISTRATION.read_text(encoding="utf-8")
        manager_registration = MANAGER_REGISTRATION.read_text(encoding="utf-8")

        self.assertNotIn("from isaaclab_tasks", tasks_source)
        self.assertNotIn("import_packages", tasks_source)
        self.assertIn('find_spec("isaaclab")', runner_source)
        self.assertNotIn("        import isaaclab", runner_source)
        self.assertNotIn("from . import agents", direct_registration)
        self.assertNotIn("from . import agents", manager_registration)


if __name__ == "__main__":
    unittest.main()

"""Franka cube-stack env with *absolute* joint-position arm action, for joint-space teleop.

Derived from ``Isaac-Stack-Cube-Franka-v0`` (stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg),
whose arm action is ``JointPositionActionCfg(scale=0.5, use_default_offset=True)`` i.e. a
*relative* command around the default pose. For a leader arm we want the action to be the
joint target itself, so scale=1 and no offset. Gripper stays binary (+1 open / -1 close)
so the stack task's observation / termination logic keeps working and record_demos.py
output stays mimic-compatible.

Registered (with a camera variant) by omy_leader_isaaclab.register() / teleop.py.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
    FrankaCubeStackEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

import os

from isaaclab.devices import DevicesCfg

from .device import OmyLeaderCfg
from .franka_config import FRANKA_HOME

ENV_ID = "Isaac-Stack-Cube-Franka-JointTeleop-v0"
CAM_ENV_ID = "Isaac-Stack-Cube-Franka-JointTeleop-Cam-v0"


@configclass
class FrankaCubeStackJointTeleopEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # stiff PD + no gravity on links, like the official IK envs; the default gains (kp=80)
        # sag ~0.25 rad under gravity, which shows up as leader/follower mismatch
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        # absolute joint targets
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, use_default_offset=False
        )
        # start exactly at the pose the retargeter assumes (J3 = 0), no randomization
        self.events.init_franka_arm_pose = EventTerm(
            func=franka_stack_events.set_default_joint_pose,
            mode="reset",
            params={"default_pose": [*FRANKA_HOME, 0.04, 0.04]},
        )
        self.events.randomize_franka_joint_state = None
        # single env, teleop rate
        self.scene.num_envs = 1
        self.decimation = 2  # sim 120 Hz -> control 60 Hz (matches OmyToFrankaConfig.dt)
        self.sim.dt = 1.0 / 120.0
        self.episode_length_s = 10000.0
        self.teleop_devices = DevicesCfg(
            devices={
                "omy_leader": OmyLeaderCfg(
                    target="franka",
                    source=os.environ.get("OMY_LEADER_SOURCE", "serial"),
                    port=os.environ.get("OMY_LEADER_PORT", "/dev/robotis_left"),
                    tcp_port=int(os.environ.get("OMY_LEADER_TCP_PORT", "5555")),
                    calib_path=os.environ.get("OMY_LEADER_CALIB", ""),
                    auto_zero=os.environ.get("OMY_LEADER_AUTO_ZERO", "0") == "1",
                    dt=self.sim.dt * self.decimation,
                )
            }
        )


def _look_at_quat_world(pos, target, up=(0.0, 0.0, 1.0)):
    """(w, x, y, z) for Isaac Lab's ``convention="world"`` camera frame (+X forward, +Z up)."""
    import numpy as np

    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross(up, f)
    left /= np.linalg.norm(left)
    u = np.cross(f, left)
    m = np.stack([f, left, u], axis=1)  # columns = camera x, y, z in world
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = (m[2, 1] - m[1, 2]) / (4.0 * w)
    y = (m[0, 2] - m[2, 0]) / (4.0 * w)
    z = (m[1, 0] - m[0, 1]) / (4.0 * w)
    return (float(w), float(x), float(y), float(z))


# View from directly behind the robot (operator's point of view), raised so the arm does not
# hide the work area; left/right match the robot's.
VIEW_CAM_POS = (-1.25, 0.0, 1.55)
VIEW_CAM_TARGET = (0.55, 0.0, 0.05)


@configclass
class FrankaCubeStackJointTeleopCamEnvCfg(FrankaCubeStackJointTeleopEnvCfg):
    """Same env plus two RGB cameras for the operator video stream (needs --enable_cameras)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.view_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/view_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=VIEW_CAM_POS, rot=_look_at_quat_world(VIEW_CAM_POS, VIEW_CAM_TARGET), convention="world"
            ),
        )
        # wrist cam: same placement as the official visuomotor stack env
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=180,
            width=240,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )
        self.sim.render.antialiasing_mode = "DLAA"


def register():
    import gymnasium as gym

    for env_id, cls in ((ENV_ID, "FrankaCubeStackJointTeleopEnvCfg"), (CAM_ENV_ID, "FrankaCubeStackJointTeleopCamEnvCfg")):
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}:{cls}"},
            disable_env_checker=True,
        )

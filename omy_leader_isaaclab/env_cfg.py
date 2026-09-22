"""cyclo_lab OMY tasks with absolute joint-position actions + the omy_leader teleop device.

cyclo_lab's OMY envs use either IK-relative actions (teleop with keyboard) or
``JointPositionAction(scale=0.3, use_default_offset=True)`` (RL). A leader arm wants the action to
*be* the joint target, so these variants set scale=1 / no offset and add ``teleop_devices``.

Registered ids (suffix ``-OMY-Leader-v0``):
    Cyclo-Pick-Place-Bottle-OMY-Leader-v0   (cyclo_lab pick_place, the IL task)
    Cyclo-Lift-Cube-OMY-Leader-v0
"""

from __future__ import annotations

import os

from isaaclab.devices import DevicesCfg
from isaaclab.envs import mdp
from isaaclab.utils import configclass

from .device import OmyLeaderCfg

# where the device reads the leader; override with env vars so record_demos.py needs no new flags
_SOURCE = os.environ.get("OMY_LEADER_SOURCE", "serial")
_PORT = os.environ.get("OMY_LEADER_PORT", "/dev/robotis_left")
_CALIB = os.environ.get("OMY_LEADER_CALIB", "")
_TCP_PORT = int(os.environ.get("OMY_LEADER_TCP_PORT", "5555"))
_AUTO_ZERO = os.environ.get("OMY_LEADER_AUTO_ZERO", "0") == "1"


def _teleop_devices(dt: float) -> DevicesCfg:
    return DevicesCfg(
        devices={
            "omy_leader": OmyLeaderCfg(
                source=_SOURCE, port=_PORT, tcp_port=_TCP_PORT, calib_path=_CALIB, auto_zero=_AUTO_ZERO, dt=dt
            )
        }
    )


def _make_joint_teleop(base_cls):
    @configclass
    class _Cfg(base_cls):
        def __post_init__(self):
            super().__post_init__()
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["joint[1-6]"], scale=1.0, use_default_offset=False
            )
            self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["rh_r1_joint"],
                open_command_expr={"rh_r1_joint": 0.0},
                close_command_expr={"rh_r1_joint": 0.8},
            )
            self.scene.num_envs = 1
            self.teleop_devices = _teleop_devices(self.sim.dt * self.decimation)

    _Cfg.__name__ = base_cls.__name__.replace("EnvCfg", "LeaderTeleopEnvCfg")
    return _Cfg


def register_envs():
    import gymnasium as gym

    from cyclo_lab.manager_based.manipulation.lift.config.omy.joint_pos_env_cfg import OMYCubeLiftEnvCfg
    from cyclo_lab.manager_based.manipulation.pick_place.config.omy.joint_pos_env_cfg import OMYBottlePickPlaceEnvCfg

    global OMYBottlePickPlaceLeaderTeleopEnvCfg, OMYCubeLiftLeaderTeleopEnvCfg
    OMYBottlePickPlaceLeaderTeleopEnvCfg = _make_joint_teleop(OMYBottlePickPlaceEnvCfg)
    OMYCubeLiftLeaderTeleopEnvCfg = _make_joint_teleop(OMYCubeLiftEnvCfg)
    for env_id, cls in (
        ("Cyclo-Pick-Place-Bottle-OMY-Leader-v0", "OMYBottlePickPlaceLeaderTeleopEnvCfg"),
        ("Cyclo-Lift-Cube-OMY-Leader-v0", "OMYCubeLiftLeaderTeleopEnvCfg"),
    ):
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}:{cls}"},
            disable_env_checker=True,
        )

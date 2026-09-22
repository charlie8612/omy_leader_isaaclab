"""L100 -> OMY (same kinematics) 1:1 joint mirror. Pure numpy.

The L100 is a scaled OMY-F3M, so the follower target is simply

    q_omy[i] = sign[i] * (raw[i] - zero[i])        i = joint1..joint6

with sign / zero from a one-time calibration (the plugin reading direction is not guaranteed to
match the URDF axis, and the encoder zero may not sit exactly on the URDF zero). Gripper: the
leader trigger reading is thresholded into a binary open/close command for
``BinaryJointPositionAction`` (+1 open / -1 close; Isaac Lab closes on ``action < 0``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

OMY_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
# cyclo_lab OMY_CFG default pose (joint1..6); used by auto-zero: hold the L100 in this pose.
OMY_DEFAULT_Q = (0.0, -1.55, 2.66, -1.1, 1.6, 0.0)
OMY_VEL_MAX = 6.0  # rad/s, OMY_CFG velocity_limit_sim


@dataclass
class MirrorConfig:
    sign: dict[str, float] = field(default_factory=lambda: {j: 1.0 for j in OMY_JOINTS})
    zero_rad: dict[str, float] = field(default_factory=lambda: {j: 0.0 for j in OMY_JOINTS})
    # trigger reading when released / fully squeezed (rad); either ordering works
    gripper_open_rad: float = 0.0
    gripper_closed_rad: float = math.radians(10.0)
    gripper_threshold: float = 0.5  # fraction of travel; hysteresis +/- 0.1 around it
    joint_limit_rad: float = math.radians(170.0)  # symmetric soft clamp (sim joints are +/-2pi)
    vel_scale: float = 1.0
    dt: float = 1.0 / 50.0

    @classmethod
    def from_json(cls, path: str) -> "MirrorConfig":
        with open(path) as f:
            c = json.load(f)
        cfg = cls()
        cfg.sign.update({k: float(v) for k, v in c.get("omy_sign", {}).items()})
        cfg.zero_rad.update({k: float(v) for k, v in c.get("mirror_zero_rad", c.get("omy_zero_rad", {})).items()})
        cfg.gripper_open_rad = float(c.get("gripper_open_rad", cfg.gripper_open_rad))
        cfg.gripper_closed_rad = float(c.get("gripper_closed_rad", cfg.gripper_closed_rad))
        return cfg


class OmyMirror:
    """Stateful: rate limits toward the leader pose and keeps gripper hysteresis."""

    def __init__(self, cfg: MirrorConfig, initial_q=OMY_DEFAULT_Q):
        self.cfg = cfg
        self._sign = np.array([cfg.sign.get(j, 1.0) for j in OMY_JOINTS])
        self._zero = np.array([cfg.zero_rad.get(j, 0.0) for j in OMY_JOINTS])
        self._dq_max = OMY_VEL_MAX * cfg.vel_scale * cfg.dt
        self._grip_closed = False
        self.reset(initial_q)

    def reset(self, q=OMY_DEFAULT_Q) -> None:
        self._q = np.array(q, dtype=float)
        self._grip_closed = False

    def raw_target(self, omy: dict[str, float]) -> np.ndarray:
        raw = np.array([omy[f"{j}.pos"] for j in OMY_JOINTS])
        return self._sign * (raw - self._zero)

    def step(self, omy: dict[str, float]) -> np.ndarray:
        """-> [q1..q6, gripper(+1 open / -1 close)]"""
        q_des = np.clip(self.raw_target(omy), -self.cfg.joint_limit_rad, self.cfg.joint_limit_rad)
        self._q = self._q + np.clip(q_des - self._q, -self._dq_max, self._dq_max)
        return np.concatenate([self._q, [self.gripper_command(omy["gripper.pos"])]])

    def gripper_command(self, grip_rad: float) -> float:
        c = self.cfg
        f = (grip_rad - c.gripper_open_rad) / (c.gripper_closed_rad - c.gripper_open_rad)  # 0 open .. 1 closed
        if self._grip_closed and f < c.gripper_threshold - 0.1:
            self._grip_closed = False
        elif not self._grip_closed and f > c.gripper_threshold + 0.1:
            self._grip_closed = True
        return -1.0 if self._grip_closed else 1.0

    def auto_zero(self, omy: dict[str, float], target=OMY_DEFAULT_Q) -> MirrorConfig:
        """Config whose zeros make the *current* leader pose map onto ``target``."""
        raw = np.array([omy[f"{j}.pos"] for j in OMY_JOINTS])
        zero = dict(self.cfg.zero_rad)
        for i, j in enumerate(OMY_JOINTS):
            zero[j] = float(raw[i] - target[i] / self._sign[i])
        cfg = MirrorConfig(**{**vars(self.cfg), "zero_rad": zero})
        return cfg

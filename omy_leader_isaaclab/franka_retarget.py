"""OMY-L100 joint radians -> Franka 7 joint targets + binary gripper. Pure numpy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .franka_config import (
    DEFAULT_OMY_TO_FRANKA_CONFIG,
    FRANKA_HOME,
    FRANKA_LOWER,
    FRANKA_UPPER,
    FRANKA_VEL_MAX,
    OmyToFrankaConfig,
)
from .franka_wrist import direct_franka_wrist, solve_franka_wrist

OMY_JOINTS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


@dataclass
class OmyToFrankaRetarget:
    """Stateful mapper. Call :meth:`step` at the control rate with the raw OMY action dict
    (``joint_1.pos`` .. ``joint_6.pos``, ``gripper.pos`` in rad, as emitted by
    ``lerobot_teleoperator_omy.OmyLeader``).

    Returns an 8-vector ``[J1..J7, gripper]`` where gripper is +1 (open) / -1 (close), which is
    the action layout of the Isaac Lab Franka stack env with ``BinaryJointPositionAction``.
    """

    config: OmyToFrankaConfig = DEFAULT_OMY_TO_FRANKA_CONFIG
    # Rate limiting starts from here; set to the env's actual joint state on reset.
    initial_q: tuple[float, ...] = FRANKA_HOME
    _q: np.ndarray = field(init=False, repr=False)
    _lo: np.ndarray = field(init=False, repr=False)
    _hi: np.ndarray = field(init=False, repr=False)
    _dq_max: np.ndarray = field(init=False, repr=False)
    _zero: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        c = self.config
        self._lo = np.array(FRANKA_LOWER) + c.limit_margin_rad
        self._hi = np.array(FRANKA_UPPER) - c.limit_margin_rad
        self._dq_max = np.array(FRANKA_VEL_MAX) * c.vel_scale * c.dt
        self._zero = np.array([c.omy_zero_rad.get(j, 0.0) for j in OMY_JOINTS])
        self._sign = np.array([c.omy_sign.get(j, 1.0) for j in OMY_JOINTS])
        self.reset(self.initial_q)

    # ------------------------------------------------------------------ public

    def reset(self, q: tuple[float, ...] | np.ndarray | None = None) -> None:
        self._q = np.array(q if q is not None else self.initial_q, dtype=float)

    def raw_target(self, omy: dict[str, float]) -> np.ndarray:
        """Unclamped, unfiltered Franka 7-vector for the given OMY reading (for calibration)."""
        c = self.config
        j = self._sign * (np.array([omy[f"{n}.pos"] for n in OMY_JOINTS]) - self._zero)
        q = np.zeros(7)
        for m in c.arm_map:
            q[m.franka_index] = m.scale * j[OMY_JOINTS.index(m.omy_joint)] + m.offset_rad
        q[2] = c.j3_fixed_rad
        if c.wrist_mode == "direct":
            q[4], q[5], q[6] = direct_franka_wrist(j[3], j[4], j[5], c.j6_reset_rad, c.j7_offset_rad)
        else:
            q[4], q[5], q[6] = solve_franka_wrist(
                j[3], j[4], j[5], c.j7_offset_rad,
                prev_j5=float(self._q[4]), prev_j7=float(self._q[6]), eps=c.wrist_singular_eps,
            )
        return q

    def step(self, omy: dict[str, float]) -> np.ndarray:
        q_des = self.raw_target(omy)
        # rate limit toward the target, then clamp to (margined) joint limits
        dq = np.clip(q_des - self._q, -self._dq_max, self._dq_max)
        self._q = np.clip(self._q + dq, self._lo, self._hi)
        return np.concatenate([self._q, [self.gripper_command(omy["gripper.pos"])]])

    def gripper_fraction(self, grip_rad: float) -> float:
        c = self.config
        f = (grip_rad - c.gripper_closed_rad) / (c.gripper_open_rad - c.gripper_closed_rad)
        return float(np.clip(f, 0.0, 1.0))

    def gripper_command(self, grip_rad: float) -> float:
        return 1.0 if self.gripper_fraction(grip_rad) >= self.config.gripper_binary_threshold else -1.0

    # ------------------------------------------------------------- calibration

    def auto_offset(self, omy: dict[str, float], target: tuple[float, ...] = FRANKA_HOME) -> OmyToFrankaConfig:
        """Return a config whose ``omy_zero_rad`` makes the *current* OMY pose map to ``target``.

        Arm: exact (three affine joints). Wrist: roll (j5) is zeroed and the pitch residual is
        put on j4 so that the L100 wrist reads as the pitch that reproduces target J6
        (for FRANKA_HOME that is j4 + j6 = pi/2, since Franka's home has the hand at 90 deg).
        Assumes target has J5 = 0 and J7 = j7_offset (true for FRANKA_HOME).
        """
        c = self.config
        raw = np.array([omy[f"{n}.pos"] for n in OMY_JOINTS])
        s = {n: c.omy_sign.get(n, 1.0) for n in OMY_JOINTS}
        zero = dict(c.omy_zero_rad)
        # j = s * (raw - zero)  ->  zero = raw - want / s
        for m in c.arm_map:
            want = (target[m.franka_index] - m.offset_rad) / m.scale
            zero[m.omy_joint] = float(raw[OMY_JOINTS.index(m.omy_joint)] - want / s[m.omy_joint])
        # wrist: READY must read (j4, j5, j6) = (0, pi/2, 0) -- the L100 working pose that maps onto
        # the Franka home wrist (see wrist.py). Assumes target has the FRANKA_HOME wrist.
        zero["joint_4"] = float(raw[3])
        zero["joint_6"] = float(raw[5])
        if c.wrist_mode == "direct":
            zero["joint_5"] = float(raw[4])
        else:
            zero["joint_5"] = float(raw[4] - (math.pi / 2) / s["joint_5"])
        from dataclasses import replace

        return replace(c, omy_zero_rad=zero)

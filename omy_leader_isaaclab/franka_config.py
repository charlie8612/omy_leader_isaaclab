"""Configuration for OMY-L100 -> Franka Panda joint-space retargeting.

Geometry (joint axes in the base frame at each robot's URDF zero config, arms pointing up):

    OMY-L100 (omy_l100_arm.urdf.xacro):  j1 Z  j2 Y  j3 Y  | j4 Y   j5 Z   j6 Y
    Franka Panda, J3 locked at 0:        J1 Z  J2 Y  J4 -Y | J5 Z   J6 -Y  J7 -Z

The arm (first three joints) has the same product-of-exponentials structure on both
robots, so J1 = j1, J2 = j2, J4 = -j3 is an exact orientation match of the forearm frame.
The wrists differ (L100 is pitch-roll-pitch, Franka is roll-pitch-roll), so J5..J7 are
solved from the L100 wrist orientation via a closed-form ZYZ decomposition (see wrist.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Classic Franka "ready" pose (0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4). J3 = 0 by construction.
FRANKA_HOME = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)

# Panda joint limits (rad), with a small safety margin applied in the retargeter.
FRANKA_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
FRANKA_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
FRANKA_VEL_MAX = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)


@dataclass(frozen=True)
class ArmJointMap:
    """Affine map for one of the three arm joints: ``franka = scale * (omy - omy_zero) + offset``."""

    franka_index: int  # 0-based index into the 7-vector
    omy_joint: str  # "joint_1" .. "joint_3"
    scale: float
    offset_rad: float = 0.0


@dataclass(frozen=True)
class OmyToFrankaConfig:
    # --- arm: exact PoE match, see module docstring ---
    arm_map: tuple[ArmJointMap, ...] = field(
        default_factory=lambda: (
            ArmJointMap(0, "joint_1", +1.0),
            ArmJointMap(1, "joint_2", +1.0),
            ArmJointMap(3, "joint_3", -1.0),
        )
    )
    j3_fixed_rad: float = 0.0

    # Per-joint zero correction applied to the raw L100 reading before anything else
    # (rad, keyed by OMY joint name). Default assumes the L100 firmware zero equals the
    # URDF zero (arm straight up, wrist straight). Fill from `publisher.py --print` if not.
    omy_zero_rad: dict[str, float] = field(
        default_factory=lambda: {f"joint_{i}": 0.0 for i in range(1, 7)}
    )
    # Per-joint direction of the plugin reading relative to the URDF axis (+1 / -1).
    # Determined once by moving each joint on the real L100 and watching the sim.
    omy_sign: dict[str, float] = field(
        default_factory=lambda: {f"joint_{i}": 1.0 for i in range(1, 7)}
    )

    # --- wrist ---
    # Constant roll between the L100 link6 frame and the Franka hand frame at the
    # "straight" configuration. pi/4 accounts for the Panda hand being mounted at -45 deg
    # on the flange; a straight L100 wrist then maps to (J5, J6, J7) = (0, pi, pi/4).
    j7_offset_rad: float = math.pi / 4
    # |sin(beta)| below this -> ZYZ singular (Franka J6 ~ 0 or ~ pi); hold J5, solve J7 only.
    wrist_singular_eps: float = 1e-3
    # "direct": fixed per-joint wrist map J6 = j6_reset + K4*j4, J5 = K5*j5, J7 = j7_offset + K7*j6
    #           (j4/j5/j6 zeroed at the reset reading; no singularity coupling)  <- default
    # "zyz"   : exact orientation match via ZYZ decomposition (couples J5/J7 near J6 = 0 / pi)
    wrist_mode: str = "direct"
    j6_reset_rad: float = math.pi

    # --- gripper ---
    # OMY handle reading (rad) at closed / open; Franka finger travel is 0 .. 0.04 m.
    gripper_closed_rad: float = -0.8
    gripper_open_rad: float = 0.0
    # Binary action threshold as a fraction of travel: below -> close (-1), else open (+1).
    gripper_binary_threshold: float = 0.5

    # --- safety ---
    limit_margin_rad: float = 0.02
    # Max joint step per call, as a fraction of FRANKA_VEL_MAX * dt. 1.0 = respect spec.
    vel_scale: float = 1.0
    dt: float = 1.0 / 60.0


DEFAULT_OMY_TO_FRANKA_CONFIG = OmyToFrankaConfig()


def load_calib(path: str, base: OmyToFrankaConfig = DEFAULT_OMY_TO_FRANKA_CONFIG) -> OmyToFrankaConfig:
    """Apply an ``omy_calib.json`` written by calib.py (signs, zero offsets, gripper endpoints)."""
    import json
    from dataclasses import replace

    with open(path) as f:
        c = json.load(f)
    return replace(
        base,
        omy_sign={k: float(v) for k, v in c["omy_sign"].items()},
        omy_zero_rad={k: float(v) for k, v in c["omy_zero_rad"].items()},
        gripper_closed_rad=float(c["gripper_closed_rad"]),
        gripper_open_rad=float(c["gripper_open_rad"]),
        wrist_mode=str(c.get("wrist_mode", base.wrist_mode)),
        j6_reset_rad=math.radians(float(c.get("j6_reset_deg", math.degrees(base.j6_reset_rad)))),
    )

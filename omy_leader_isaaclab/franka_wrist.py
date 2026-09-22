"""Closed-form wrist retargeting: L100 (Y-Z-Y) -> Franka (Z-Y-Z with J3 locked).

All rotations are expressed in "base-aligned" frames at the URDF zero configuration,
using the product-of-exponentials convention: the orientation of the tool relative to
the forearm is the product of the wrist joint rotations about their zero-config axes.

L100 wrist (axes j4 +Y, j5 +Z, j6 +Y; link6 frame == base frame at zero). The gripper sticks
out of link6 sideways along -Y (rh_r1_joint origin (0, -0.125, -0.026)), i.e. the tool axis is
parallel to j6, so j6 spins the tool and the wrist is only non-singular away from j5 = 0. The
operator's working region is j5 ~ +90 deg, where j4 = pitch, j5 = roll about the forearm,
j6 = spin about the tool. Tool frame: z along -Y of link6, x along +X, i.e. R_t = Rx(pi/2):

    R_L = Ry(j4) Rz(j5) Ry(j6) Rx(pi/2)

Franka wrist (axes J5 +Z, J6 -Y, J7 -Z; link7/flange frame at zero = Rx(pi); the Panda
hand is mounted at Rz(-pi/4) on the flange):

    R_F = Rz(J5) Ry(-J6) Rz(-J7) Rx(pi) Rz(-pi/4)

Requiring R_F == R_L R_align, with the constant R_align chosen so that the L100 working pose
(j4, j5, j6) = (0, pi/2, 0) (tool along +X of the forearm frame, like the Franka hand at its
home wrist) maps onto Franka (J5, J6, J7) = (0, pi/2, pi/4), and simplifying gives

    Rz(-J5) Ry(J6) Rz(J7 - j7_offset) = Rx(pi) R_L R_align =: M

so a ZYZ Euler decomposition of M with beta in [0, pi] yields J5 = -alpha, J6 = beta,
J7 = gamma + j7_offset. beta >= 0 is exactly Franka's one-sided J6 range.
"""

from __future__ import annotations

import math

import numpy as np


def rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_RX_PI = rx(math.pi)
_R_T = rx(math.pi / 2)  # link6 -> tool frame (tool z along link6 -Y)


def omy_wrist_rotation(j4: float, j5: float, j6: float) -> np.ndarray:
    """Tool orientation relative to the L100 forearm frame."""
    return ry(j4) @ rz(j5) @ ry(j6) @ _R_T


# Constant so that L100 (0, pi/2, 0) <-> Franka (0, pi/2, pi/4):  Rx(pi) R_L(0,pi/2,0) R_align == Ry(pi/2)
_R_ALIGN = (_RX_PI @ omy_wrist_rotation(0.0, math.pi / 2, 0.0)).T @ ry(math.pi / 2)


def franka_wrist_rotation(j5: float, j6: float, j7: float, j7_offset: float) -> np.ndarray:
    """Franka hand orientation relative to its forearm frame, in the same convention as
    :func:`omy_wrist_rotation` (i.e. already multiplied by Rz(pi)^-1 so the two compare
    directly). Used by tests to round-trip the solver."""
    m = rz(-j5) @ ry(j6) @ rz(j7 - j7_offset)  # == Rx(pi) R_L R_align
    return _RX_PI @ m @ _R_ALIGN.T


def zyz_from_matrix(m: np.ndarray, prev_alpha: float | None, eps: float) -> tuple[float, float, float]:
    """Decompose m = Rz(alpha) Ry(beta) Rz(gamma) with beta in [0, pi].

    At the singularities (beta ~ 0 or ~ pi) only alpha +/- gamma is determined; we then hold
    alpha at ``prev_alpha`` (or 0) and put the remainder into gamma.
    """
    c_beta = float(np.clip(m[2, 2], -1.0, 1.0))
    s_beta = math.hypot(m[0, 2], m[1, 2])  # >= 0 branch
    beta = math.atan2(s_beta, c_beta)

    if s_beta < eps:
        alpha = 0.0 if prev_alpha is None else prev_alpha
        if c_beta > 0.0:  # beta ~ 0: m ~ Rz(alpha + gamma)
            total = math.atan2(m[1, 0], m[0, 0])
            gamma = total - alpha
        else:  # beta ~ pi: m ~ Rz(alpha - gamma) Ry(pi)
            diff = math.atan2(-m[1, 0], -m[0, 0])
            gamma = alpha - diff
        return alpha, beta, _wrap(gamma)

    alpha = math.atan2(m[1, 2], m[0, 2])
    gamma = math.atan2(m[2, 1], -m[2, 0])
    return alpha, beta, gamma


def solve_franka_wrist(
    j4: float,
    j5: float,
    j6: float,
    j7_offset: float,
    prev_j5: float | None = None,
    prev_j7: float | None = None,
    eps: float = 1e-3,
) -> tuple[float, float, float]:
    """Map L100 wrist angles to Franka (J5, J6, J7). Angles in rad, unclamped.

    J7 is unwrapped to the 2*pi-equivalent closest to ``prev_j7`` so the operator never sees a
    sudden flip; the caller clamps to joint limits.
    """
    m = _RX_PI @ omy_wrist_rotation(j4, j5, j6) @ _R_ALIGN
    prev_alpha = None if prev_j5 is None else -prev_j5
    alpha, beta, gamma = zyz_from_matrix(m, prev_alpha, eps)
    J5 = -alpha
    J6 = beta
    J7 = gamma + j7_offset
    if prev_j7 is not None:
        J7 = prev_j7 + _wrap(J7 - prev_j7)
    return J5, J6, J7


def _wrap(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------- direct mode
# Fixed per-joint wrist mapping (no orientation solve, hence no singularity coupling):
#   J6 = j6_reset + K4 * j4      J5 = K5 * j5      J7 = j7_offset + K7 * j6
# with j4/j5/j6 measured from the reset reading. The K signs are the local directions of the ZYZ
# solution at the L100 working pose, so the feel is identical to zyz mode near reset.
K4, K5, K7 = -1.0, 1.0, -1.0


def direct_franka_wrist(j4: float, j5: float, j6: float, j6_reset: float, j7_offset: float) -> tuple[float, float, float]:
    return K5 * j5, j6_reset + K4 * j4, j7_offset + K7 * j6

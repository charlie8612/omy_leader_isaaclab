import math

import numpy as np

from omy_leader_isaaclab.mirror import OMY_DEFAULT_Q, OMY_JOINTS, MirrorConfig, OmyMirror


def reading(q, grip=0.0):
    d = {f"{j}.pos": v for j, v in zip(OMY_JOINTS, q)}
    d["gripper.pos"] = grip
    return d


def test_identity_mirror_and_gripper_hysteresis():
    m = OmyMirror(MirrorConfig(vel_scale=1e9))
    a = m.step(reading(OMY_DEFAULT_Q))
    assert np.allclose(a[:6], OMY_DEFAULT_Q) and a[6] == 1.0
    assert m.step(reading(OMY_DEFAULT_Q, math.radians(9)))[6] == -1.0  # squeezed -> close
    assert m.step(reading(OMY_DEFAULT_Q, math.radians(5)))[6] == -1.0  # inside hysteresis band: stays closed
    assert m.step(reading(OMY_DEFAULT_Q, math.radians(3)))[6] == 1.0   # released -> open


def test_sign_and_auto_zero():
    cfg = MirrorConfig(sign={**{j: 1.0 for j in OMY_JOINTS}, "joint_2": -1.0, "joint_4": -1.0})
    m = OmyMirror(cfg, initial_q=OMY_DEFAULT_Q)
    raw = [0.3, 1.0, 2.0, 0.5, 1.2, -0.4]
    cfg2 = m.auto_zero(reading(raw))
    m2 = OmyMirror(cfg2)
    assert np.allclose(m2.raw_target(reading(raw)), OMY_DEFAULT_Q, atol=1e-9)
    # moving joint_2 raw *down* must move the target *up* (sign -1)
    r2 = list(raw); r2[1] -= 0.1
    assert m2.raw_target(reading(r2))[1] > OMY_DEFAULT_Q[1]


def test_rate_limit():
    m = OmyMirror(MirrorConfig(dt=0.02), initial_q=OMY_DEFAULT_Q)
    a = m.step(reading([2.0, *OMY_DEFAULT_Q[1:]]))
    assert abs(a[0] - OMY_DEFAULT_Q[0]) <= 6.0 * 0.02 + 1e-9

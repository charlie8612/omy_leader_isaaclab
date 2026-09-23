"""Pure-numpy checks for the OMY-L100 -> Franka retargeter (no Isaac Lab needed)."""

import math

import numpy as np
import pytest

from omy_leader_isaaclab.franka_config import FRANKA_HOME, OmyToFrankaConfig
from omy_leader_isaaclab.franka_retarget import OmyToFrankaRetarget
from omy_leader_isaaclab.franka_wrist import franka_wrist_rotation, omy_wrist_rotation, ry, rz, solve_franka_wrist

J7_OFF = math.pi / 4


def omy(j1=0, j2=0, j3=0, j4=0, j5=0, j6=0, grip=0.0):
    return {
        "joint_1.pos": j1, "joint_2.pos": j2, "joint_3.pos": j3,
        "joint_4.pos": j4, "joint_5.pos": j5, "joint_6.pos": j6, "gripper.pos": grip,
    }


def test_working_pose_maps_to_franka_home_wrist():
    # L100 working pose (j4, j5, j6) = (0, pi/2, 0) <-> Franka home wrist (0, pi/2, pi/4)
    J5, J6, J7 = solve_franka_wrist(0, math.pi / 2, 0, J7_OFF)
    assert np.allclose([J5, J6, J7], [0.0, math.pi / 2, J7_OFF], atol=1e-9)


def test_wrist_joint_correspondence_near_working_pose():
    # around the working pose: j6 spins the tool (-> J7 only), j4 pitches (-> J6 only), j5 rolls (-> J5 only)
    base = np.array(solve_franka_wrist(0, math.pi / 2, 0, J7_OFF))
    d = 0.2
    dj6 = np.array(solve_franka_wrist(0, math.pi / 2, d, J7_OFF)) - base
    dj4 = np.array(solve_franka_wrist(d, math.pi / 2, 0, J7_OFF)) - base
    dj5 = np.array(solve_franka_wrist(0, math.pi / 2 + d, 0, J7_OFF)) - base
    assert abs(abs(dj6[2]) - d) < 1e-9 and np.allclose(dj6[:2], 0, atol=1e-9), dj6
    assert abs(abs(dj4[1]) - d) < 1e-9 and abs(dj4[0]) < 1e-9 and abs(dj4[2]) < 1e-9, dj4
    assert abs(abs(dj5[0]) - d) < 1e-9 and abs(dj5[1]) < 1e-9 and abs(dj5[2]) < 1e-9, dj5


@pytest.mark.parametrize("seed", range(200))
def test_wrist_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    j4, j5, j6 = rng.uniform(-2.5, 2.5, 3)
    J5, J6, J7 = solve_franka_wrist(j4, j5, j6, J7_OFF)
    assert 0.0 <= J6 <= math.pi
    R_l = omy_wrist_rotation(j4, j5, j6)
    R_f = franka_wrist_rotation(J5, J6, J7, J7_OFF)
    assert np.allclose(R_l, R_f, atol=1e-9), f"{seed}: orientation mismatch"


def test_wrist_singularity_holds_j5():
    # tool along the forearm (+Z): L100 (j4, j5) = (-pi/2, pi/2) -> Franka J6 = pi, J5/J7 coupled; J5 sticks to prev
    J5, J6, J7 = solve_franka_wrist(-math.pi / 2, math.pi / 2, 0.3, J7_OFF, prev_j5=0.3, prev_j7=J7_OFF)
    assert abs(J6 - math.pi) < 1e-6 and abs(J5 - 0.3) < 1e-9
    R_l = omy_wrist_rotation(-math.pi / 2, math.pi / 2, 0.3)
    assert np.allclose(R_l, franka_wrist_rotation(J5, J6, J7, J7_OFF), atol=1e-9)


def test_j7_unwraps_toward_prev():
    prev = 2.7
    J5, J6, J7 = solve_franka_wrist(0.5, 0.2, 0.3, J7_OFF, prev_j5=0.0, prev_j7=prev)
    assert abs(J7 - prev) <= math.pi


def test_arm_map_and_j3_locked():
    rt = OmyToFrankaRetarget(OmyToFrankaConfig(vel_scale=1e9, wrist_mode="zyz"))  # no rate limit
    a = rt.step(omy(j1=0.4, j2=-0.3, j3=1.2, j5=math.pi / 2))
    assert np.allclose(a[:4], [0.4, -0.3, 0.0, -1.2], atol=1e-9)
    assert a[7] == 1.0  # gripper open at 0 rad


def test_rate_limit_and_clamp():
    cfg = OmyToFrankaConfig(dt=1 / 60, wrist_mode="zyz")
    rt = OmyToFrankaRetarget(cfg, initial_q=FRANKA_HOME)
    a = rt.step(omy(j1=2.0, j5=math.pi / 2))
    # one step can move at most vel_max*dt on J1
    assert abs(a[0] - FRANKA_HOME[0]) <= 2.175 / 60 + 1e-9
    for _ in range(2000):
        a = rt.step(omy(j1=5.0, j5=math.pi / 2))
    assert a[0] == pytest.approx(2.8973 - cfg.limit_margin_rad)


def test_gripper_binary():
    rt = OmyToFrankaRetarget()
    assert rt.gripper_command(math.radians(20)) == -1.0
    assert rt.gripper_command(0.0) == 1.0
    assert rt.gripper_command(math.radians(9)) == 1.0 and rt.gripper_command(math.radians(11)) == -1.0


def test_auto_offset_maps_current_pose_to_home():
    rt = OmyToFrankaRetarget(OmyToFrankaConfig(vel_scale=1e9, wrist_mode="zyz"))
    cur = omy(j1=0.2, j2=-1.0, j3=2.0, j4=1.0, j5=0.3, j6=0.2)
    cfg2 = rt.auto_offset(cur)
    rt2 = OmyToFrankaRetarget(cfg2)
    q = rt2.raw_target(cur)
    assert np.allclose(q, FRANKA_HOME, atol=1e-6)


def test_direct_wrist_mode():
    rt = OmyToFrankaRetarget(OmyToFrankaConfig(vel_scale=1e9, wrist_mode="direct"))
    q = rt.raw_target(omy())
    assert np.allclose(q[4:], [0.0, math.pi, math.pi / 4])            # reset wrist
    q = rt.raw_target(omy(j4=0.2, j5=0.3, j6=0.4))
    assert np.allclose(q[4:], [0.3, math.pi - 0.2, math.pi / 4 - 0.4])  # K5=+1, K4=-1, K7=-1, no coupling


def test_scale_and_offset():
    cfg = OmyToFrankaConfig(vel_scale=1e9, wrist_mode="direct",
                            omy_scale={"joint_1": 1.0, "joint_2": 1.5, "joint_3": 2.0, "joint_4": 1.0, "joint_5": 1.0, "joint_6": 1.0},
                            franka_offset_rad=(0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0))
    rt = OmyToFrankaRetarget(cfg)
    q = rt.raw_target(omy(j2=0.2, j3=0.3))
    assert np.allclose(q[:4], [0.0, 0.3 + 0.1, 0.0, -0.6 + 0.2])   # scale then offset; J4 = -j3
    # auto_offset still maps the current pose to home with scales applied
    cfg2 = rt.auto_offset(omy(j1=0.2, j2=-1.0, j3=2.0, j4=1.0, j5=0.3, j6=0.2))
    q2 = OmyToFrankaRetarget(cfg2).raw_target(omy(j1=0.2, j2=-1.0, j3=2.0, j4=1.0, j5=0.3, j6=0.2))
    assert np.allclose(q2[:4], np.array(FRANKA_HOME[:4]) + np.array([0.0, 0.1, 0.0, 0.2]), atol=1e-6)

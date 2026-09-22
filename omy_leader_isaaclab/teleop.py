"""Standalone runner: the OMY-L100 drives a simulated OMY (cyclo_lab) or Franka (Isaac Lab) in joint space.

    cd <IsaacLab>
    ./isaaclab.sh -p <repo>/omy_leader_isaaclab/teleop.py --target omy                    # L100 on this host
    ./isaaclab.sh -p <repo>/omy_leader_isaaclab/teleop.py --target franka --headless --stream
    ... --source tcp                      # leader on another host: publisher.py there + ssh -R 5555
    ... --calib omy_calib.json | --auto-zero   # zero the leader (hold the target's default pose 3 s)
    ... --task <env id>                   # any env with absolute JointPositionAction + BinaryJointPositionAction

--stream renders a third-person camera (and the wrist cam on Franka) and serves JPEG frames on
--video-port for viewer.py (ssh -L 5556:localhost:5556 from the laptop). For recording demos use
the task framework's own script (e.g. cyclo_lab record_demos.py --teleop_device omy_leader).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_TASK = {"omy": "Cyclo-Lift-Cube-OMY-Leader-v0", "franka": "Isaac-Stack-Cube-Franka-JointTeleop-v0"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--target", choices=["omy", "franka"], default="omy")
parser.add_argument("--task", default="", help=f"env id (default per target: {DEFAULT_TASK})")
parser.add_argument("--source", choices=["serial", "tcp"], default="serial")
parser.add_argument("--port", default="/dev/robotis_left")
parser.add_argument("--tcp-port", type=int, default=5555)
parser.add_argument("--calib", default="")
parser.add_argument("--auto-zero", action="store_true")
parser.add_argument("--steps", type=int, default=0, help="exit after N steps (smoke tests)")
parser.add_argument("--stream", action="store_true", help="serve JPEG video on --video-port (implies --enable_cameras)")
parser.add_argument("--video-port", type=int, default=5556)
parser.add_argument("--video-every", type=int, default=2)
parser.add_argument("--no-realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.stream:
    args.enable_cameras = True
task = args.task or DEFAULT_TASK[args.target]
if args.target == "franka" and args.stream and task == DEFAULT_TASK["franka"]:
    task = "Isaac-Stack-Cube-Franka-JointTeleop-Cam-v0"
# the env cfgs read the leader settings from env vars (so third-party record scripts need no new flags)
os.environ["OMY_LEADER_SOURCE"] = args.source
os.environ["OMY_LEADER_PORT"] = args.port
os.environ["OMY_LEADER_TCP_PORT"] = str(args.tcp_port)
os.environ["OMY_LEADER_CALIB"] = args.calib
os.environ["OMY_LEADER_AUTO_ZERO"] = "1" if args.auto_zero else "0"

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cyclo_lab  # noqa: E402, F401  (OMY tasks; optional)
except ImportError:
    pass
import omy_leader_isaaclab  # noqa: E402, F401  (registers device + envs)
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

VIEW_CAM_POS, VIEW_CAM_TARGET = (-1.25, 0.0, 1.55), (0.55, 0.0, 0.05)


def _rgb(cam) -> np.ndarray:
    return cam.data.output["rgb"][0][..., :3].to(torch.uint8).cpu().numpy()


def main():
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    video = None
    if args.stream:
        from omy_leader_isaaclab.video import VideoServer, encode_jpeg

        if not hasattr(cfg.scene, "view_cam"):
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import CameraCfg
            from omy_leader_isaaclab.franka_env_cfg import _look_at_quat_world

            cfg.scene.view_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/view_cam", update_period=0.0, height=480, width=640, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)),
                offset=CameraCfg.OffsetCfg(pos=VIEW_CAM_POS, rot=_look_at_quat_world(VIEW_CAM_POS, VIEW_CAM_TARGET), convention="world"),
            )
        video = VideoServer(port=args.video_port)
        print(f"[teleop] video on 127.0.0.1:{args.video_port} (laptop: ssh -L {args.video_port}:localhost:{args.video_port} <host>; python -m omy_leader_isaaclab.viewer)")
    env = gym.make(task, cfg=cfg).unwrapped
    device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
    print(device)
    env.reset()
    device.reset()
    n_arm = 7 if args.target == "franka" else 6
    step_dt = cfg.sim.dt * cfg.decimation
    n, t_next = 0, time.perf_counter()
    while app.is_running():
        with torch.inference_mode():
            a = device.advance()
            env.step(a.unsqueeze(0).to(env.device))
            n += 1
            if video is not None and n % args.video_every == 0:
                img = _rgb(env.scene["view_cam"])
                if "wrist_cam" in env.scene.keys():
                    w = _rgb(env.scene["wrist_cam"])
                    img = img.copy()
                    img[8 : 8 + w.shape[0], 8 : 8 + w.shape[1]] = w
                video.push(encode_jpeg(img))
            if n % 120 == 0:
                q = env.scene["robot"].data.joint_pos[0, :n_arm].cpu().numpy()
                cmd = a[:n_arm].cpu().numpy()
                print(f"[teleop] step {n}  cmd={cmd.round(2)} grip={a[n_arm]:+.0f}  q-cmd={(q - cmd).round(3)}")
            if args.steps and n >= args.steps:
                break
        if not args.no_realtime:
            t_next += step_dt
            d = t_next - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                t_next = time.perf_counter()
    device.close()
    env.close()


if __name__ == "__main__":
    main()
    app.close()

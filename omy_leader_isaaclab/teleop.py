"""Standalone runner (no recording): OMY-L100 drives a cyclo_lab OMY task, optional JPEG video stream.

    cd <IsaacLab>; ./isaaclab.sh -p <repo>/omy_leader_isaaclab/teleop.py --task Cyclo-Lift-Cube-OMY-Leader-v0 \
        --source tcp --headless --stream          # leader via publisher on another host + video on 5556
    ... --source serial --port /dev/robotis_left   # leader plugged into this machine
    ... --calib omy_calib.json  |  --auto-zero     # hold the OMY default pose at start to zero

For recording use cyclo_lab's own script instead (same device, env vars pick the leader source):
    OMY_LEADER_SOURCE=tcp OMY_LEADER_CALIB=omy_calib.json \
    python scripts/imitation_learning/isaaclab_recorder/record_demos.py --task Cyclo-Pick-Place-Bottle-OMY-Leader-v0 --teleop_device omy_leader ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Cyclo-Lift-Cube-OMY-Leader-v0")
parser.add_argument("--source", choices=["serial", "tcp"], default="serial")
parser.add_argument("--port", default="/dev/robotis_left")
parser.add_argument("--tcp-port", type=int, default=5555)
parser.add_argument("--calib", default="")
parser.add_argument("--auto-zero", action="store_true")
parser.add_argument("--steps", type=int, default=0, help="exit after N steps (smoke tests)")
parser.add_argument("--stream", action="store_true", help="JPEG video of the viewport camera on --video-port")
parser.add_argument("--video-port", type=int, default=5556)
parser.add_argument("--no-realtime", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.stream:
    args.enable_cameras = True
# the env cfg reads the leader settings from env vars (so record_demos.py needs no new flags)
os.environ["OMY_LEADER_SOURCE"] = args.source
os.environ["OMY_LEADER_PORT"] = args.port
os.environ["OMY_LEADER_TCP_PORT"] = str(args.tcp_port)
os.environ["OMY_LEADER_CALIB"] = args.calib
os.environ["OMY_LEADER_AUTO_ZERO"] = "1" if args.auto_zero else "0"

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cyclo_lab  # noqa: E402, F401
import omy_leader_isaaclab  # noqa: E402, F401  (registers device + envs)
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    video = None
    if args.stream:
        from isaaclab.sensors import CameraCfg  # noqa: PLC0415
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from omy_leader_isaaclab.video import VideoServer, encode_jpeg  # noqa: PLC0415

        cfg.scene.view_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/view_cam", update_period=0.0, height=480, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)),
            offset=CameraCfg.OffsetCfg(pos=(-1.25, 0.0, 1.55), rot=(0.9063, 0.0, 0.4226, 0.0), convention="world"),
        )
        video = VideoServer(port=args.video_port)
    env = gym.make(args.task, cfg=cfg).unwrapped
    device = create_teleop_device("omy_leader", cfg.teleop_devices.devices, {})
    print(device)
    env.reset()
    device.reset()
    step_dt = cfg.sim.dt * cfg.decimation
    n, t_next = 0, time.perf_counter()
    while app.is_running():
        with torch.inference_mode():
            a = device.advance()
            env.step(a.unsqueeze(0).to(env.device))
            n += 1
            if video is not None and n % 2 == 0:
                img = env.scene["view_cam"].data.output["rgb"][0][..., :3].to(torch.uint8).cpu().numpy()
                video.push(encode_jpeg(img))
            if n % 100 == 0:
                q = env.scene["robot"].data.joint_pos[0, :6].cpu().numpy()
                print(f"[teleop] step {n}  cmd={a[:6].cpu().numpy().round(2)} grip={a[6]:+.0f}  q-cmd={(q - a[:6].cpu().numpy()).round(3)}")
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

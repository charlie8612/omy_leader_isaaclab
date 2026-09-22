"""Hardware-free stand-in for the L100: streams the OMY default pose with a slow wiggle over TCP.

    python -m omy_leader_isaaclab.fake_publisher [--tcp-port 5555] [--amp 0.3]
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "omy_leader_isaaclab"

import argparse
import math
import time

from .link import DEFAULT_PORT, FrameServer, pack
from .mirror import OMY_DEFAULT_Q, OMY_JOINTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcp-host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--amp", type=float, default=0.3)
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--still", action="store_true")
    args = ap.parse_args()
    server = FrameServer(args.tcp_host, args.tcp_port)
    print(f"[fake-omy] listening on {args.tcp_host}:{args.tcp_port}")
    seq, t0 = 0, time.time()
    while True:
        t = time.time() - t0
        w = 0.0 if args.still else 2 * math.pi * t / args.period
        a = {f"{j}.pos": q for j, q in zip(OMY_JOINTS, OMY_DEFAULT_Q)}
        a["joint_1.pos"] += args.amp * math.sin(w)
        a["joint_3.pos"] += 0.5 * args.amp * math.sin(0.5 * w)
        a["joint_5.pos"] += 0.5 * args.amp * math.sin(0.7 * w)
        a["joint_6.pos"] += args.amp * math.sin(0.3 * w)
        a["gripper.pos"] = math.radians(10) if (not args.still and int(t / 3) % 2) else 0.0
        server.send(pack(seq, a))
        seq += 1
        if seq % int(args.fps * 5) == 0:
            print(f"[fake-omy] seq={seq} clients={server.n_clients}")
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()

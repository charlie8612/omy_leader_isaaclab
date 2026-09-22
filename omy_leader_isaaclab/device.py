"""Isaac Lab teleop device for the ROBOTIS OMY-L100 leader arm.

Plugs into Isaac Lab's device factory (``create_teleop_device``) so cyclo_lab's ``record_demos.py``
uses it with no changes:

    env_cfg.teleop_devices = DevicesCfg(devices={"omy_leader": OmyLeaderCfg(source="serial", port=...)})
    record_demos.py --task <task> --teleop_device omy_leader

Output: 7-vector ``[joint1..joint6 (rad), gripper (+1 open / -1 close)]`` for an env whose actions are
``JointPositionActionCfg(joint_names=["joint[1-6]"], scale=1, use_default_offset=False)`` +
``BinaryJointPositionActionCfg(rh_r1_joint)`` (see env_cfg.py).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from isaaclab.devices import DeviceBase, DeviceCfg

from .mirror import OMY_DEFAULT_Q, MirrorConfig, OmyMirror


@dataclass
class OmyLeaderCfg(DeviceCfg):
    source: str = "serial"  # "serial" (DynamixelSDK on this host) | "tcp" (publisher.py on another host)
    port: str = "/dev/robotis_left"
    baudrate: int = 4_000_000
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 5555
    calib_path: str = ""  # omy_calib.json (sign / zero / gripper); empty = defaults (+1, 0)
    # if True, when the stream comes up: hold the leader in OMY's default pose, zeros are computed
    auto_zero: bool = False
    auto_zero_settle_s: float = 3.0
    stale_s: float = 0.5  # hold the last command if no fresh reading for this long
    dt: float = 1.0 / 50.0  # env control period, for the rate limiter
    vel_scale: float = 1.0
    hz: float = 100.0  # serial read rate
    # keyboard keys are not available here; record_demos wires callbacks by name ("R", "START"...)
    callbacks: dict = field(default_factory=dict)


class OmyLeaderDevice(DeviceBase):
    def __init__(self, cfg: OmyLeaderCfg, retargeters=None):
        super().__init__(retargeters)
        self.cfg = cfg
        mcfg = MirrorConfig.from_json(cfg.calib_path) if cfg.calib_path else MirrorConfig()
        mcfg.dt, mcfg.vel_scale = cfg.dt, cfg.vel_scale
        self._mirror = OmyMirror(mcfg)
        self._callbacks: dict[str, Callable] = {}
        self._last = np.concatenate([OMY_DEFAULT_Q, [1.0]])
        self._warned = False
        self._zeroed = not cfg.auto_zero
        self._t_up = None
        if cfg.source == "tcp":
            from .link import FrameClient

            self._client = FrameClient(cfg.tcp_host, cfg.tcp_port)
            self._latest = self._client.latest
        elif cfg.source == "serial":
            from .omy_serial import OmySerialLeader

            self._serial = OmySerialLeader(cfg.port, cfg.baudrate)
            self._buf = (None, 0.0)
            import threading

            self._stop = threading.Event()
            threading.Thread(target=self._serial_loop, daemon=True).start()
            self._latest = lambda: (self._buf[0], time.time() - self._buf[1])
        else:
            raise ValueError(f"unknown source {cfg.source!r}")

    def _serial_loop(self):
        period = 1.0 / self.cfg.hz
        while not self._stop.is_set():
            try:
                self._buf = (self._serial.read(), time.time())
            except Exception as e:  # noqa: BLE001
                print(f"[omy_leader] serial read error: {e}")
                time.sleep(0.5)
            time.sleep(period)

    # ---- DeviceBase -----------------------------------------------------------------
    def __str__(self) -> str:
        return (
            f"OMY-L100 leader ({self.cfg.source}) -> OMY joint mirror\n"
            "  joints 1..6 : 1:1 (sign / zero from calib)\n"
            "  gripper     : squeeze trigger -> close\n"
            "  keys        : none (use record_demos' START/STOP/RESET via other means or a keyboard device)\n"
        )

    def reset(self) -> None:
        self._mirror.reset()

    def add_callback(self, key, func: Callable) -> None:
        self._callbacks[key] = func

    def _get_raw_data(self):
        return self._latest()

    def advance(self) -> torch.Tensor:
        a, age = self._latest()
        if a is None or age > self.cfg.stale_s:
            if not self._warned:
                print(f"[omy_leader] no fresh leader data (age {age:.2f}s) - holding last command")
                self._warned = True
            return torch.as_tensor(self._last, dtype=torch.float32, device=self.cfg.sim_device)
        self._warned = False
        if not self._zeroed:
            if self._t_up is None:
                self._t_up = time.time()
                print(f"[omy_leader] hold the L100 in the OMY default pose - zeroing in {self.cfg.auto_zero_settle_s:.0f}s")
                return torch.as_tensor(self._last, dtype=torch.float32, device=self.cfg.sim_device)
            if time.time() - self._t_up < self.cfg.auto_zero_settle_s:
                return torch.as_tensor(self._last, dtype=torch.float32, device=self.cfg.sim_device)
            self._mirror = OmyMirror(self._mirror.auto_zero(a))
            self._zeroed = True
            print("[omy_leader] auto-zero:", {k: round(v, 3) for k, v in self._mirror.cfg.zero_rad.items()})
        self._last = self._mirror.step(a)
        return torch.as_tensor(self._last, dtype=torch.float32, device=self.cfg.sim_device)

    def close(self):
        if self.cfg.source == "serial":
            self._stop.set()
            self._serial.close()
        else:
            self._client.close()


# Register with Isaac Lab's factory so env_cfg.teleop_devices can name this device.
try:
    from isaaclab.devices import teleop_device_factory as _f

    if hasattr(_f, "DEVICE_MAP"):
        _f.DEVICE_MAP[OmyLeaderCfg] = OmyLeaderDevice
    OmyLeaderCfg.class_type = OmyLeaderDevice  # Isaac Lab >= 3.0 resolves via cfg.class_type
except Exception:  # pragma: no cover
    pass

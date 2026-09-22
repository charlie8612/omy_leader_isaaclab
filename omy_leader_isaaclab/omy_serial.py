"""Minimal OMY-L100 reader on DynamixelSDK (Protocol 2.0, X-series), no lerobot dependency.

Motors: joint_1..3 XH540-W150 (ids 1-3), joint_4..6 XC330-T288 (ids 4-6), gripper XC330-T181 (id 7).
Arm motors are read torque-off in Extended Position mode; the gripper trigger is put in
current-based position mode with a small current limit so it springs back to ``gripper_open_pos``
(same behaviour as ``lerobot_teleoperator_omy``). Angles are returned in radians with the X-series
centre tick (2048) as zero, so they are directly comparable with the lerobot plugin's output.
"""

from __future__ import annotations

import math
import time

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_DRIVE_MODE = 10
ADDR_CURRENT_LIMIT = 38
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
MODE_EXTENDED_POSITION = 4
MODE_CURRENT_POSITION = 5
TICKS_PER_REV = 4096
CENTER = 2048

JOINT_IDS = {"joint_1": 1, "joint_2": 2, "joint_3": 3, "joint_4": 4, "joint_5": 5, "joint_6": 6, "gripper": 7}


def ticks_to_rad(ticks: int) -> float:
    return (ticks - CENTER) * (2.0 * math.pi / TICKS_PER_REV)


class OmySerialLeader:
    def __init__(
        self,
        port: str = "/dev/robotis_left",
        baudrate: int = 4_000_000,
        gripper_spring: bool = True,
        gripper_open_ticks: int = int(CENTER + 48.2 / 200 * TICKS_PER_REV),  # == plugin gripper_open_pos 48.2
        gripper_current_limit: int = 100,
    ):
        from dynamixel_sdk import GroupSyncRead, PacketHandler, PortHandler  # noqa: PLC0415

        self._port = PortHandler(port)
        if not self._port.openPort():
            raise ConnectionError(f"cannot open {port}")
        if not self._port.setBaudRate(baudrate):
            raise ConnectionError(f"cannot set baudrate {baudrate} on {port}")
        self._ph = PacketHandler(2.0)
        self._ids = list(JOINT_IDS.values())
        self._reader = GroupSyncRead(self._port, self._ph, ADDR_PRESENT_POSITION, 4)
        for i in self._ids:
            self._reader.addParam(i)
        # configure
        for i in self._ids:
            self._write1(i, ADDR_TORQUE_ENABLE, 0)
        for name, i in JOINT_IDS.items():
            if name != "gripper":
                self._write1(i, ADDR_OPERATING_MODE, MODE_EXTENDED_POSITION)
        g = JOINT_IDS["gripper"]
        if gripper_spring:
            self._write1(g, ADDR_OPERATING_MODE, MODE_CURRENT_POSITION)
            self._write2(g, ADDR_CURRENT_LIMIT, gripper_current_limit)
            self._write2(g, ADDR_GOAL_CURRENT, gripper_current_limit)
            self._write1(g, ADDR_TORQUE_ENABLE, 1)
            self._write4(g, ADDR_GOAL_POSITION, gripper_open_ticks)
        self._gripper_open_ticks = gripper_open_ticks

    # ---- packet helpers ------------------------------------------------------------
    def _check(self, res, err, what):
        if res != 0 or err != 0:
            raise ConnectionError(f"{what}: {self._ph.getTxRxResult(res)} {self._ph.getRxPacketError(err)}")

    def _write1(self, i, addr, v):
        self._check(*self._ph.write1ByteTxRx(self._port, i, addr, v), f"write1 id{i}@{addr}")

    def _write2(self, i, addr, v):
        self._check(*self._ph.write2ByteTxRx(self._port, i, addr, v), f"write2 id{i}@{addr}")

    def _write4(self, i, addr, v):
        self._check(*self._ph.write4ByteTxRx(self._port, i, addr, v), f"write4 id{i}@{addr}")

    # ---- public -----------------------------------------------------------------
    def read(self) -> dict[str, float]:
        """{'joint_1.pos': rad, ..., 'gripper.pos': rad (0 = released)}"""
        if self._reader.txRxPacket() != 0:
            raise ConnectionError("sync read failed")
        out = {}
        for name, i in JOINT_IDS.items():
            v = self._reader.getData(i, ADDR_PRESENT_POSITION, 4)
            if v >= 2**31:  # signed 32-bit (extended position)
                v -= 2**32
            out[f"{name}.pos"] = ticks_to_rad(v)
        out["gripper.pos"] -= ticks_to_rad(self._gripper_open_ticks)
        return out

    def close(self):
        try:
            self._write1(JOINT_IDS["gripper"], ADDR_TORQUE_ENABLE, 0)
        finally:
            self._port.closePort()


class SerialClient:
    """Background reader with the same ``latest()`` API as :class:`link.FrameClient`, so calibration
    and monitor tools can read the leader directly on the sim machine (the common single-PC setup)."""

    def __init__(self, port: str = "/dev/robotis_left", baudrate: int = 4_000_000, hz: float = 100.0):
        import threading

        self._leader = OmySerialLeader(port, baudrate)
        self._buf: tuple[dict | None, float] = (None, 0.0)
        self._stop = threading.Event()
        self._period = 1.0 / hz
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._buf = (self._leader.read(), time.time())
            except Exception as e:  # noqa: BLE001
                print(f"[omy_serial] read error: {e}")
                time.sleep(0.5)
            time.sleep(self._period)

    def latest(self):
        a, t = self._buf
        return (None, float("inf")) if a is None else (a, time.time() - t)

    def close(self):
        self._stop.set()
        self._leader.close()


def open_reader(source: str = "serial", port: str = "/dev/robotis_left", baudrate: int = 4_000_000,
                tcp_host: str = "127.0.0.1", tcp_port: int = 5555):
    """Return an object with ``latest()`` for either the local serial leader or a remote publisher."""
    if source == "tcp":
        from .link import FrameClient

        return FrameClient(tcp_host, tcp_port)
    return SerialClient(port, baudrate)


if __name__ == "__main__":  # quick check: python -m omy_leader_isaaclab.omy_serial /dev/robotis_left
    import sys

    l = OmySerialLeader(sys.argv[1] if len(sys.argv) > 1 else "/dev/robotis_left")
    try:
        while True:
            a = l.read()
            print("  ".join(f"{k[:-4]}={math.degrees(v):+7.1f}" for k, v in a.items()), end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        l.close()

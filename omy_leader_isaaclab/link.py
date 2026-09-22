"""Tiny TCP link for streaming OMY joint readings across hosts (stdlib only).

TCP rather than UDP because the L100 host and the sim host sit behind different NATs and
the hop is an ``ssh -R`` reverse tunnel, which only forwards TCP. Frames are fixed-size so
the receiver never has to parse; it just keeps the latest one.

Frame (little-endian, 72 bytes):
    double t_send      publisher wall clock (time.time())
    uint32 seq
    uint32 _pad
    double[7]          joint_1..joint_6, gripper   (rad)
"""

from __future__ import annotations

import socket
import struct
import threading
import time

FRAME = struct.Struct("<dII7d")
FRAME_SIZE = FRAME.size  # 72
JOINT_KEYS = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
DEFAULT_PORT = 5555


def pack(seq: int, action: dict[str, float]) -> bytes:
    return FRAME.pack(time.time(), seq & 0xFFFFFFFF, 0, *(action[f"{k}.pos"] for k in JOINT_KEYS))


def unpack(buf: bytes) -> tuple[float, int, dict[str, float]]:
    t, seq, _, *vals = FRAME.unpack(buf)
    return t, seq, {f"{k}.pos": v for k, v in zip(JOINT_KEYS, vals)}


class FrameServer:
    """Accepts any number of clients and fans out frames (used by publisher.py)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(4)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while True:
            c, _ = self._srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(c)

    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def send(self, frame: bytes) -> None:
        with self._lock:
            dead = []
            for c in self._clients:
                try:
                    c.sendall(frame)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)
                c.close()


class FrameClient:
    """Background reader that keeps only the newest frame (used by device.py)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, reconnect_s: float = 1.0):
        self._addr = (host, port)
        self._reconnect_s = reconnect_s
        self._latest: tuple[float, int, dict[str, float]] | None = None
        self._t_recv = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection(self._addr, timeout=2.0) as s:
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    s.settimeout(1.0)
                    buf = b""
                    while not self._stop.is_set():
                        chunk = s.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                        # keep only the last complete frame
                        n = len(buf) // FRAME_SIZE
                        if n:
                            frame = buf[(n - 1) * FRAME_SIZE : n * FRAME_SIZE]
                            buf = buf[n * FRAME_SIZE :]
                            with self._lock:
                                self._latest = unpack(frame)
                                self._t_recv = time.time()
            except (OSError, socket.timeout):
                time.sleep(self._reconnect_s)

    def latest(self) -> tuple[dict[str, float] | None, float]:
        """(action dict or None, age in seconds since last frame arrived)."""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest[2], time.time() - self._t_recv

    def close(self):
        self._stop.set()

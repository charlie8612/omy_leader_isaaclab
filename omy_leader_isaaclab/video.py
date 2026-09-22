"""JPEG-over-TCP video stream from the sim host to the operator's laptop (stdlib + cv2/PIL).

Why not Isaac's WebRTC livestream: it needs UDP reachability to the sim host, which sits behind
the lab NAT; the only thing that gets through is ssh. So we render a camera in the sim, JPEG each
frame and push it over a TCP socket that the laptop reaches via ``ssh -L 5556:localhost:5556``.

Wire format per frame:  header ``<dII`` (t_send, seq, n_bytes) then ``n_bytes`` of JPEG.
Each client has its own sender thread holding only the *latest* frame, so a slow link drops frames
instead of stalling the sim loop.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

HDR = struct.Struct("<dII")
DEFAULT_VIDEO_PORT = 5556


def encode_jpeg(rgb, quality: int = 80) -> bytes:
    """rgb: HxWx3 uint8 numpy array."""
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
    except ImportError:
        import io

        from PIL import Image

        bio = io.BytesIO()
        Image.fromarray(rgb).save(bio, format="JPEG", quality=quality)
        return bio.getvalue()


def decode_jpeg(data: bytes):
    """-> HxWx3 uint8 RGB numpy array."""
    try:
        import cv2
        import numpy as np

        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        import io

        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class VideoServer:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_VIDEO_PORT):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(4)
        self._latest: bytes | None = None
        self._seq = 0
        self._cv = threading.Condition()
        self._n_clients = 0
        threading.Thread(target=self._accept_loop, daemon=True).start()

    @property
    def n_clients(self) -> int:
        return self._n_clients

    def push(self, jpeg: bytes) -> None:
        with self._cv:
            self._seq += 1
            self._latest = HDR.pack(time.time(), self._seq, len(jpeg)) + jpeg
            self._cv.notify_all()

    def _accept_loop(self):
        while True:
            c, _ = self._srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._sender, args=(c,), daemon=True).start()

    def _sender(self, c: socket.socket):
        self._n_clients += 1
        sent_seq = -1
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._latest is not None and self._seq != sent_seq, timeout=1.0)
                    if self._latest is None or self._seq == sent_seq:
                        continue
                    frame, sent_seq = self._latest, self._seq
                c.sendall(frame)
        except OSError:
            pass
        finally:
            self._n_clients -= 1
            c.close()


class VideoClient:
    """Blocking reader; ``recv()`` returns (t_send, seq, jpeg_bytes)."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_VIDEO_PORT):
        self._s = socket.create_connection((host, port), timeout=10.0)
        self._s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _read(self, n: int) -> bytes:
        buf = b""
        waited = 0.0
        while len(buf) < n:
            try:
                chunk = self._s.recv(n - len(buf))
            except (socket.timeout, TimeoutError):
                # the sim only pushes frames once its control loop runs (i.e. once the leader stream is
                # up); keep waiting instead of failing so the viewer can be started first
                waited += 10.0
                print(f"[viewer] no frames yet ({waited:.0f}s) - sim waiting for the leader stream?", flush=True)
                continue
            if not chunk:
                raise ConnectionError("video stream closed")
            buf += chunk
        return buf

    def recv(self) -> tuple[float, int, bytes]:
        t, seq, n = HDR.unpack(self._read(HDR.size))
        return t, seq, self._read(n)

    def close(self):
        self._s.close()

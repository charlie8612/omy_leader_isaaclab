"""Operator-side viewer for the sim camera stream. Runs on the Mac; needs numpy + (cv2 or PIL).

    ssh -N -L 5556:localhost:5556 <sim host> &        # only when the sim runs elsewhere
    omy-leader-viewer                 # window; q / Esc quits
    omy-leader-viewer --save out.jpg --frames 30   # headless check: save 30th frame
"""

from __future__ import annotations

if __package__ in (None, ""):  # allow `python omy_leader_isaaclab/<file>.py` as well as `python -m omy_leader_isaaclab.<file>`
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "omy_leader_isaaclab"

import argparse
import time

from .video import DEFAULT_VIDEO_PORT, VideoClient, decode_jpeg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_VIDEO_PORT)
    ap.add_argument("--save", help="save a frame to this path instead of opening a window")
    ap.add_argument("--frames", type=int, default=30, help="with --save: which frame to save")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    args = ap.parse_args()

    client = VideoClient(args.host, args.port)
    print(f"[viewer] connected to {args.host}:{args.port}")

    if args.save:
        for i in range(args.frames):
            t, seq, jpg = client.recv()
        with open(args.save, "wb") as f:
            f.write(jpg)
        print(f"[viewer] saved seq={seq} ({len(jpg)} bytes, one-way age {time.time() - t:.3f}s) -> {args.save}")
        return

    try:
        import cv2

        n, t0 = 0, time.time()
        while True:
            t, seq, jpg = client.recv()
            rgb = decode_jpeg(jpg)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale)
            n += 1
            if n % 30 == 0:
                fps = n / (time.time() - t0)
                cv2.setWindowTitle("franka", f"franka teleop  {fps:.0f} fps  age {1000 * (time.time() - t):.0f} ms")
            cv2.imshow("franka", bgr)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except ImportError:
        import tkinter as tk

        from PIL import Image, ImageTk

        root = tk.Tk()
        root.title("franka teleop")
        label = tk.Label(root)
        label.pack()
        root.bind("<Key-q>", lambda e: root.destroy())
        root.bind("<Escape>", lambda e: root.destroy())

        def tick():
            t, seq, jpg = client.recv()
            img = Image.fromarray(decode_jpeg(jpg))
            if args.scale != 1.0:
                img = img.resize((int(img.width * args.scale), int(img.height * args.scale)))
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo)
            label.image = photo
            root.title(f"franka teleop  age {1000 * (time.time() - t):.0f} ms")
            root.after(1, tick)

        root.after(1, tick)
        root.mainloop()
    finally:
        client.close()


if __name__ == "__main__":
    main()

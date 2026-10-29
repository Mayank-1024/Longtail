"""Writes tests/fixtures/sam31_SYNTHETIC.txt and its clip index.

SYNTHETIC: hand-built in the documented wire format, not a real SAM response.

Run from the repo root: uv run python tests/fixtures/make_sam_fixture.py
Replace with a real recorded response from scripts/sam_smoke.py once there is a Meta key.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1])]

from sam_wire import frame_line, record  # noqa: E402

from longtail.vision.clips import ClipCrop, ClipFrame, ClipIndex  # noqa: E402

W, H = 64, 48


def rect(x1, y1, x2, y2):
    m = np.zeros((H, W), bool)
    m[y1 : y2 + 1, x1 : x2 + 1] = True
    return m


def main() -> None:
    lines = [
        "SYNTHETIC fixture in the SAM 3.1 wire format; not a real SAM response (this line is free text).",
        frame_line(0, [record(0, rect(5, 10, 14, 19), conf=0.9), record(1, rect(30, 5, 39, 9))]),
        frame_line(
            1, [record(0, rect(7, 10, 16, 19), conf=0.85), record(1, rect(31, 5, 40, 9), mask_first=True)]
        ),
        # 16 objects in one frame: SAM's cap, so the response is marked truncated.
        frame_line(2, [record(k, rect(4 * k % 60, 30, 4 * k % 60 + 2, 33), conf=0.5) for k in range(16)]),
        # Object 0 reported at half the clip's resolution (w=32, h=24): boxes must scale back up.
        frame_line(3, [record(0, rect(8, 10, 17, 19), conf=0.8, scale=0.5)]),
    ]
    (HERE / "sam31_SYNTHETIC.txt").write_text("\n".join(lines) + "\n")
    index = ClipIndex(
        clip_id="SYNTHETIC",
        scene_id="synthetic-fixture",
        camera="front_camera",
        fps=10.0,
        codec="h264",
        pix_fmt="yuv420p",
        crop=ClipCrop(name="full", x0=0, y0=0, width=W, height=H, full_width=W, full_height=H),
        frames=[ClipFrame(clip_frame=k, frame_idx=10 + k, ts=100.0 + 0.1 * k) for k in range(4)],
    )
    (HERE / "sam31_SYNTHETIC.clip_index.json").write_text(index.model_dump_json(indent=1) + "\n")


if __name__ == "__main__":
    main()

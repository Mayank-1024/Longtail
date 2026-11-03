"""Scene frames to an H.264 clip for SAM 3.1 (plan C3), with an index from clip frame to dataset time.

One MP4 per (scene, camera, crop): yuv420p at a constant frame rate equal to the scene's native rate, checked
after encoding for its frame count and for no dropped or duplicated frames. Clips are cached by content hash.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from longtail.data.base import Scene

Crop = Literal["full", "left", "right"]
CLIP_VERSION = 1  # bump to invalidate cached clips when the encoding changes
# Lossless enough for segmentation, fast to encode; a fixed GOP keeps decoding simple.
FFMPEG_VIDEO_ARGS = [
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-g",
    "10",
]


class ClipCrop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Crop
    x0: int  # offset of the clip's left edge in the full camera image, pixels
    y0: int
    width: int  # clip size, pixels
    height: int
    full_width: int  # camera image size, pixels
    full_height: int


class ClipFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_frame: int  # 0-based frame number in the MP4 (SAM's <Nf>)
    frame_idx: int  # camera frame index in the scene
    ts: float  # camera frame timestamp, seconds


class ClipIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str  # content hash
    scene_id: str
    camera: str
    fps: float
    codec: str
    pix_fmt: str
    crop: ClipCrop
    frames: list[ClipFrame]

    def frame(self, clip_frame: int) -> ClipFrame:
        if not 0 <= clip_frame < len(self.frames):
            raise IndexError(f"clip frame {clip_frame} outside a {len(self.frames)}-frame clip")
        return self.frames[clip_frame]


@dataclass(frozen=True)
class Clip:
    path: Path
    index: ClipIndex
    cached: bool  # served from the cache rather than encoded now


def cache_root() -> Path:
    # ASSUMPTION A-112: caches live under $LONGTAIL_CACHE_DIR (default ./cache, git-ignored)
    return Path(os.environ.get("LONGTAIL_CACHE_DIR", "cache"))


def crop_box(width: int, height: int, crop: Crop) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) of the crop, with even sizes for yuv420p."""
    # ASSUMPTION A-111: optional half-image crops (left, right) for SAM's 16-object cap; boxes map back
    even_w, even_h = width - width % 2, height - height % 2
    if crop == "full":
        return 0, 0, even_w, even_h
    half = (width // 2) - (width // 2) % 2
    if crop == "left":
        return 0, 0, half, even_h
    if crop == "right":
        return width - half, 0, half, even_h
    raise ValueError(f"unknown crop {crop!r}")


def native_fps(ts: np.ndarray) -> float:
    """The scene's frame rate from its timestamps (10 Hz for PandaSet)."""
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
    if not step > 0:
        raise ValueError(f"timestamps do not increase (median step {step}); the frame rate is unknown")
    return float(round(1.0 / step, 3))


def build_clip(scene: Scene, camera: str, crop: Crop = "full", cache_dir: Path | None = None) -> Clip:
    """Encode (or fetch from the cache) one camera's frames as a constant-rate H.264 MP4."""
    frames = scene.frames(camera)
    if not frames:
        raise ValueError(f"{scene.scene_id} {camera}: no frames")
    for f in frames:
        if not f.image_path.is_file():
            raise FileNotFoundError(
                f"{scene.scene_id} {camera} frame {f.frame_idx}: no image at {f.image_path}"
            )
    full_w, full_h = scene.image_size(camera)
    x0, y0, w, h = crop_box(full_w, full_h, crop)
    fps = native_fps(np.array([f.ts for f in frames]))

    digest = hashlib.sha256()
    meta = [CLIP_VERSION, scene.scene_id, camera, crop, fps, FFMPEG_VIDEO_ARGS]
    digest.update(json.dumps(meta).encode())
    for f in frames:
        digest.update(f"{f.frame_idx}:{f.ts:.6f}:".encode())
        digest.update(hashlib.sha256(f.image_path.read_bytes()).digest())
    clip_id = digest.hexdigest()[:24]

    out = (cache_dir or cache_root() / "clips") / f"{clip_id}.mp4"
    index_path = out.with_suffix(".clip_index.json")
    if out.is_file() and index_path.is_file():
        return Clip(out, ClipIndex.model_validate_json(index_path.read_text()), cached=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        *FFMPEG_VIDEO_ARGS,
        "-fps_mode",
        "cfr",
        "-r",
        f"{fps}",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            with Image.open(f.image_path) as img:
                rgb = np.asarray(img.convert("RGB"))
            if rgb.shape[:2] != (full_h, full_w):
                raise ValueError(f"{f.image_path}: {rgb.shape[1]}x{rgb.shape[0]}, expected {full_w}x{full_h}")
            proc.stdin.write(np.ascontiguousarray(rgb[y0 : y0 + h, x0 : x0 + w]).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')}")
    check_clip(tmp, len(frames), fps, (w, h))
    tmp.rename(out)

    index = ClipIndex(
        clip_id=clip_id,
        scene_id=scene.scene_id,
        camera=camera,
        fps=fps,
        codec="h264",
        pix_fmt="yuv420p",
        crop=ClipCrop(name=crop, x0=x0, y0=y0, width=w, height=h, full_width=full_w, full_height=full_h),
        frames=[ClipFrame(clip_frame=k, frame_idx=f.frame_idx, ts=f.ts) for k, f in enumerate(frames)],
    )
    index_path.write_text(index.model_dump_json(indent=1))
    return Clip(out, index, cached=False)


def check_clip(path: Path, expected_frames: int, fps: float, size: tuple[int, int]) -> None:
    """Decode the clip and fail unless it has exactly the expected frames, evenly spaced at fps."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height:frame=pts_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    times = np.array([float(fr["pts_time"]) for fr in info.get("frames", [])])
    problems = []
    if (stream["codec_name"], stream["pix_fmt"]) != ("h264", "yuv420p"):
        problems.append(f"codec {stream['codec_name']}/{stream['pix_fmt']}")
    if (stream["width"], stream["height"]) != tuple(size):
        problems.append(f"size {stream['width']}x{stream['height']}, expected {size[0]}x{size[1]}")
    if len(times) != expected_frames:
        problems.append(f"{len(times)} decoded frames, expected {expected_frames}")
    elif len(times) > 1 and not np.allclose(np.diff(times), 1.0 / fps, atol=0.25 / fps):
        problems.append("frame times are not evenly spaced (a dropped or duplicated frame)")
    if problems:
        raise RuntimeError(f"{path}: " + "; ".join(problems))

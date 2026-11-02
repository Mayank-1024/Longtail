"""Save LiDAR-on-camera overlays for all six cameras at 3 frames of one PandaSet scene (plan C2 check).

    uv run python scripts/overlay.py [--scene 002] [--frames 10 40 70]

Points are coloured by camera depth (red near, blue far). Images go to runs/overlays/<scene>/, which
is git-ignored: dataset frames never enter the repo. Also prints the camera-to-LiDAR gap distribution.

Without --frames, the three frames are picked where the most labelled objects are moving.
"""

# ASSUMPTION A-45: frame picking reads labels inside eval_ground_truth(); only frame numbers leave the guard

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from longtail.data.guard import eval_ground_truth
from longtail.data.index import pandaset_root
from longtail.data.pandaset import CAMERA_ORDER, PandaSetScene, list_scenes
from longtail.geometry.project import project_sweep
from longtail.geometry.sync import format_gap_distribution, gap_distribution, sync_camera

FAR_M = 50.0
MOSAIC = (
    ("front_left_camera", "front_camera", "front_right_camera"),
    ("left_camera", "back_camera", "right_camera"),
)


def moving_counts(scene: PandaSetScene) -> np.ndarray:
    """Labelled objects in motion per frame. Labels stay inside the guard; only counts come out."""
    counts = np.zeros(scene.meta.n_frames, dtype=int)
    with eval_ground_truth():
        for i in range(scene.meta.n_frames):
            df = scene.labels(i)
            flags = np.zeros(len(df), dtype=bool)
            for col, value in (
                ("attributes.object_motion", "Moving"),
                ("attributes.pedestrian_behavior", "Walking"),
            ):
                if col in df:
                    flags |= (df[col] == value).to_numpy()
            counts[i] = int(flags.sum())
    return counts


def pick_frames(counts: np.ndarray, n: int = 3) -> list[int]:
    """n frames spread across the scene, each the busiest in its third."""
    return [int(c[np.argmax(counts[c])]) for c in np.array_split(np.arange(len(counts)), n)]


def depth_colours(depth: np.ndarray) -> np.ndarray:
    """Red (near) through yellow and green to blue (far), as uint8 RGB."""
    x = np.clip(depth / FAR_M, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 0.5), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 3.5), 0, 1)
    return (np.column_stack([r, g, b]) * 255).astype(np.uint8)


def draw(image: Image.Image, pixels: np.ndarray, depth: np.ndarray, radius: int = 1) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    order = np.argsort(-depth)  # far first, so near points end on top
    px, colours = pixels[order], depth_colours(depth[order])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            u, v = px[:, 0] + dx, px[:, 1] + dy
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            arr[v[ok], u[ok]] = colours[ok]
    return Image.fromarray(arr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=pandaset_root())
    ap.add_argument("--scene")
    ap.add_argument("--frames", type=int, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("runs/overlays"))
    args = ap.parse_args(argv)

    scenes = list_scenes(args.root)
    if not scenes:
        print(f"no PandaSet sequences under {args.root}", file=sys.stderr)
        return 1
    scene = PandaSetScene(args.root, args.scene or scenes[0])
    if args.frames:
        frames = args.frames
    else:
        counts = moving_counts(scene)
        frames = pick_frames(counts)
        print(f"moving objects per picked frame: {dict(zip(frames, counts[frames].tolist(), strict=True))}")

    out = args.out / scene.scene_id
    out.mkdir(parents=True, exist_ok=True)
    synced = {cam: sync_camera(scene, cam) for cam in scene.cameras}
    lidar_ts = scene.lidar_timestamps()
    for i in frames:
        tiles = {}
        for cam in [c for c in CAMERA_ORDER if c in scene.cameras]:
            frame = scene.frames(cam)[i]
            s = synced[cam][i]
            sweep = scene.lidar(lidar_ts[s.lidar_idx])
            p = project_sweep(scene, sweep, cam, frame)
            with Image.open(frame.image_path) as img:
                tile = draw(img, p.pixels(), p.depth)
            path = out / f"{i:02d}_{cam}.jpg"
            tile.save(path, quality=90)
            tiles[cam] = tile
            flag = "" if s.synced else "  UNSYNCED"
            print(f"{path}: {len(p)} points, sweep {s.lidar_idx}, gap {s.gap_s * 1000:+.1f} ms{flag}")
        w, h = tiles["front_camera"].size
        mosaic = Image.new("RGB", (3 * w // 2, 2 * h // 2))
        for r, row in enumerate(MOSAIC):
            for c, cam in enumerate(row):
                mosaic.paste(tiles[cam].resize((w // 2, h // 2)), (c * w // 2, r * h // 2))
        mosaic.save(out / f"{i:02d}_mosaic.jpg", quality=85)
        print(f"{out / f'{i:02d}_mosaic.jpg'}")

    print(f"\ncamera-to-LiDAR gaps, scene {scene.scene_id}")
    print(format_gap_distribution(gap_distribution(scene)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

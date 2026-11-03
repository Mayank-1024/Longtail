"""Fetch parts of PandaSet straight out of the Hugging Face zip, without downloading the whole 44.5 GB.

    uv run python scripts/fetch_pandaset.py --parts labels                  # all 103 scenes, labels only
    uv run python scripts/fetch_pandaset.py --parts full --scenes 001,002   # everything for two scenes
    uv run python scripts/fetch_pandaset.py --list                          # sizes per scene and part

The zip's central directory is read with HTTP range requests, then each wanted member is fetched with one
range request and inflated locally. Files land at $LONGTAIL_DATA_DIR/pandaset/<scene>/..., the layout the
loader reads (A-33). Files already present with the right size are skipped, so a run can be resumed.

Parts:
- labels: cuboid labels, GPS, and the timestamps, poses and intrinsics JSON of the LiDAR and every camera;
  enough for ground truth from labels (C7) and the zero-positive check, about 0.3 GB for all 103 scenes
- cameras: the camera JPEGs (about 0.1 GB per scene)
- lidar: the LiDAR sweeps (about 0.33 GB per scene)
- full: all of the above, plus semantic segmentation where the scene has it
"""

import argparse
import io
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ASSUMPTION A-227: the Hugging Face mirror georghess/pandaset is one zip with members pandaset/<scene>/...,
# and its CDN answers byte-range requests (checked 2026-11-03: 103 scenes, 75,758 members, 44.5 GB)
URL = "https://huggingface.co/datasets/georghess/pandaset/resolve/main/pandaset.zip"
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
LOCAL_SLACK = 1024  # the local header's extra field can differ in length from the central directory's
RETRIES = 4


class Remote:
    """Byte ranges of the zip; the signed CDN link is re-resolved when it expires."""

    def __init__(self, url: str = URL) -> None:
        self.url = url
        self._lock = threading.Lock()
        self._cdn = ""
        self.size = 0
        self._resolve()

    def _resolve(self) -> None:
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
            self._cdn = r.url

    def get(self, start: int, end: int) -> bytes:
        """Bytes start..end inclusive, clipped to the file."""
        end = min(end, self.size - 1)
        for attempt in range(RETRIES + 1):
            req = urllib.request.Request(self._cdn, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == RETRIES:
                    raise
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
                    with self._lock:
                        self._resolve()
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


class _RangeFile(io.RawIOBase):
    """A seekable file over Remote, enough for zipfile to read the central directory."""

    def __init__(self, remote: Remote) -> None:
        self.remote, self.pos = remote, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.remote.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.remote.size or len(b) == 0:
            return 0
        data = self.remote.get(self.pos, self.pos + len(b) - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def members(remote: Remote) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BufferedReader(_RangeFile(remote), buffer_size=1 << 20)) as z:
        return [i for i in z.infolist() if not i.is_dir()]


def part_of(name: str) -> str | None:
    """Which part a member belongs to: labels, cameras, lidar, semseg; None outside a scene folder."""
    p = name.split("/")
    if len(p) < 4 or p[0] != "pandaset":
        return None
    if p[2] == "annotations":
        return "labels" if p[3] == "cuboids" else "semseg"
    if p[2] == "meta" or name.endswith(".json"):
        return "labels"
    return {"camera": "cameras", "lidar": "lidar"}.get(p[2])


def wanted(parts: str) -> set[str]:
    return {"labels", "cameras", "lidar", "semseg"} if parts == "full" else {"labels", parts}


def fetch_member(remote: Remote, info: zipfile.ZipInfo, root: Path) -> int:
    out = root / info.filename
    if out.is_file() and out.stat().st_size == info.file_size:
        return 0
    head = info.header_offset
    blob = remote.get(head, head + LOCAL_HEADER.size + len(info.filename) + LOCAL_SLACK + info.compress_size)
    sig, *_, name_len, extra_len = LOCAL_HEADER.unpack_from(blob)
    if sig != b"PK\x03\x04":
        raise ValueError(f"no local header for {info.filename}")
    start = LOCAL_HEADER.size + name_len + extra_len
    raw = blob[start : start + info.compress_size]
    if len(raw) < info.compress_size:  # an unusually long extra field: fetch the rest
        rest = remote.get(head + start + len(raw), head + start + info.compress_size - 1)
        raw += rest
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = raw
    else:
        raise ValueError(f"{info.filename}: unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"{info.filename}: CRC mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return len(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", choices=["labels", "cameras", "lidar", "full"], default="labels")
    ap.add_argument("--scenes", default="all", help="comma-separated scene ids, or 'all'")
    ap.add_argument("--data-dir", default=os.environ.get("LONGTAIL_DATA_DIR", "./data"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", action="store_true", help="print sizes per scene and part, fetch nothing")
    args = ap.parse_args()

    remote = Remote()
    infos = members(remote)
    if args.list:
        sizes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in infos:
            if (part := part_of(i.filename)) is not None:
                sizes[i.filename.split("/")[1]][part] += i.compress_size
        print("scene  " + "  ".join(f"{p:>8}" for p in ("labels", "cameras", "lidar", "semseg")))
        for scene in sorted(sizes):
            row = sizes[scene]
            mb = [row.get(p, 0) / 1e6 for p in ("labels", "cameras", "lidar", "semseg")]
            print(f"{scene:5}  " + "  ".join(f"{x:7.1f}M" for x in mb))
        return 0

    scenes = None if args.scenes == "all" else set(args.scenes.split(","))
    keep = wanted(args.parts)
    todo = [
        i
        for i in infos
        if part_of(i.filename) in keep and (scenes is None or i.filename.split("/")[1] in scenes)
    ]
    if scenes is not None and (missing := scenes - {i.filename.split("/")[1] for i in todo}):
        print(f"unknown scene ids: {sorted(missing)}", file=sys.stderr)
        return 1
    total = sum(i.compress_size for i in todo)
    print(f"{len(todo)} files, {total / 1e9:.2f} GB compressed, into {args.data_dir}/pandaset")
    root = Path(args.data_dir)
    done = fetched = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_member, remote, i, root) for i in todo]
        for f in as_completed(futures):
            fetched += f.result()
            done += 1
            if done % 500 == 0 or done == len(todo):
                rate = fetched / max(time.time() - t0, 1e-6) / 1e6
                print(f"  {done}/{len(todo)} files, {fetched / 1e9:.2f} GB new, {rate:.1f} MB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

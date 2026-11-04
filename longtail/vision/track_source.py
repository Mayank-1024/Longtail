"""The real TrackSource for verify() (A-86): SAM 3.1 tracks for a scene, camera and concept."""

import functools
import gzip
import hashlib
import json
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from longtail.contracts import Track2D
from longtail.data.base import Scene
from longtail.verify.config import load_verify_config
from longtail.vision.clips import Clip, build_clip, cache_root
from longtail.vision.sam_client import SamClient, transport_from_env
from longtail.vision.sam_parse import parse

# ASSUMPTION A-118: "halves" sends the left and right halves as two clips; right-half object ids are offset by
# 1000 so the halves' tracks never collide; an object across the split may yield one track per half
RIGHT_HALF_ID_OFFSET = 1000

# ASSUMPTION A-229: parsed tracks are cached on disk by response text, clip index, concept and parser version,
# because meta-sam-parser decodes masks in pure Python (about 80 s per 80-frame camera clip)
TRACK_CACHE_VERSION = "1"  # bump when parse() changes what it returns


def _tracks_path(text: str, clip: Clip, concept: str) -> Path:
    h = hashlib.sha256()
    for part in (
        TRACK_CACHE_VERSION,
        version("meta-sam-parser"),
        concept,
        clip.index.model_dump_json(),
        text,
    ):
        h.update(part.encode())
        h.update(b"\0")
    return cache_root() / "sam_tracks" / f"{h.hexdigest()[:32]}.json.gz"


def cached_parse(text: str, clip: Clip, concept: str) -> list[Track2D]:
    """parse(), once per response: later calls (other runs or processes) read the saved tracks."""
    path = _tracks_path(text, clip, concept)
    if path.is_file():
        return [Track2D.model_validate(t) for t in json.loads(gzip.decompress(path.read_bytes()))]
    tracks = parse(text, clip.index, concept)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{id(tracks)}.tmp")
    tmp.write_bytes(gzip.compress(json.dumps([t.model_dump(mode="json") for t in tracks]).encode(), 3))
    tmp.replace(path)
    return tracks


def _capped(tracks: list[Track2D]) -> list[Track2D]:
    """v2 (A-248): every track of a response with truncated_min_objects distinct objects is truncated."""
    n = load_verify_config().association.truncated_min_objects
    # ASSUMPTION A-248: (v2) SAM's cap is per response, so 16 distinct objects mean later ones may be missing
    if n is None or len({t.obj_id for t in tracks}) < n:
        return tracks
    return [t.model_copy(update={"truncated": True}) for t in tracks]


def _tracks(
    client: SamClient,
    scene: Scene,
    camera: str,
    concept: str,
    crop: Literal["full", "halves"],
    clip_cache: Path | None,
) -> list[Track2D]:
    if crop == "full":
        clip = build_clip(scene, camera, "full", clip_cache)
        return _capped(cached_parse(client.segment(clip, concept).text, clip, concept))
    tracks = []
    for half, offset in (("left", 0), ("right", RIGHT_HALF_ID_OFFSET)):
        clip = build_clip(scene, camera, half, clip_cache)
        for t in _capped(cached_parse(client.segment(clip, concept).text, clip, concept)):
            tracks.append(t.model_copy(update={"obj_id": t.obj_id + offset}))
    return tracks


def warm(
    scene_id: str,
    camera: str,
    concept: str,
    *,
    crop: Literal["full", "halves"] = "full",
    clip_cache: Path | None = None,
    replay: bool | None = None,
) -> int:
    """Make the SAM call and parse it into the caches, in a worker process; returns the number of tracks.

    Opens the scene by id from the dataset root, so it serves the real source only (not synthetic scenes).
    """
    from longtail.data.index import open_scene

    client = SamClient(transport_from_env(), replay=replay)
    return len(_tracks(client, open_scene(scene_id), camera, concept, crop, clip_cache))


def sam_track_source(
    client: SamClient,
    *,
    cameras: Iterable[str] | None = None,
    crop: Literal["full", "halves"] = "full",
    clip_cache: Path | None = None,
):
    """A TrackSource: (scene, camera, concept) -> Track2Ds from SAM 3.1, in full-image pixels.

    `source.warm(scene_id, camera, concept)` fills the same caches from another process (pipeline prefetch).
    """
    allowed = set(cameras) if cameras is not None else None

    def source(scene: Scene, camera: str, concept: str) -> list[Track2D]:
        if allowed is not None and camera not in allowed:
            return []
        return _tracks(client, scene, camera, concept, crop, clip_cache)

    source.cameras = allowed  # type: ignore[attr-defined]
    source.warm = functools.partial(warm, crop=crop, clip_cache=clip_cache, replay=client.replay)  # type: ignore[attr-defined]
    return source

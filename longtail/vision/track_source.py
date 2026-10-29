"""The real TrackSource for verify() (A-86): SAM 3.1 tracks for a scene, camera and concept."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from longtail.contracts import Track2D
from longtail.data.base import Scene
from longtail.vision.clips import build_clip
from longtail.vision.sam_client import SamClient
from longtail.vision.sam_parse import parse

# ASSUMPTION A-118: "halves" sends the left and right halves as two clips; right-half object ids are offset by
# 1000 so the halves' tracks never collide; an object across the split may yield one track per half
RIGHT_HALF_ID_OFFSET = 1000


def sam_track_source(
    client: SamClient,
    *,
    cameras: Iterable[str] | None = None,
    crop: Literal["full", "halves"] = "full",
    clip_cache: Path | None = None,
):
    """A TrackSource: (scene, camera, concept) -> Track2Ds from SAM 3.1, in full-image pixels."""
    allowed = set(cameras) if cameras is not None else None

    def source(scene: Scene, camera: str, concept: str) -> list[Track2D]:
        if allowed is not None and camera not in allowed:
            return []
        if crop == "full":
            clip = build_clip(scene, camera, "full", clip_cache)
            return parse(client.segment(clip, concept).text, clip.index, concept)
        tracks = []
        for half, offset in (("left", 0), ("right", RIGHT_HALF_ID_OFFSET)):
            clip = build_clip(scene, camera, half, clip_cache)
            for t in parse(client.segment(clip, concept).text, clip.index, concept):
                tracks.append(t.model_copy(update={"obj_id": t.obj_id + offset}))
        return tracks

    return source

"""SAM 3.1 video output to Track2D (plan C3), on Meta's meta-sam-parser.

The wire format (plan 3.1, and the parser's documentation): one line per frame, ``<Nf>`` then comma-separated
records ``id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..[;c=..]|><|mask;x=0;y=0;data=H,W,PAYLOAD[;c=..]|>``.
Boxes are inclusive pixels of a w x h source; masks are box-local rasters. Here they become inclusive boxes
and COCO RLE masks in full camera-image pixels (A-20), mapped back through the clip's crop.
"""

# ASSUMPTION A-11: SAM 3.1 output follows plan 3.1, parsed by meta-sam-parser 0.1.0; the smoke test confirms

import re
from collections import defaultdict

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_rle,
    video_segmentation_format,
)

from longtail.contracts import Frame2D, MaskRLE, Track2D
from longtail.vision import rle
from longtail.vision.clips import ClipIndex

# Plan 3.1: at most 16 tracked objects per frame; a frame at the cap may be missing objects.
MAX_OBJECTS_PER_FRAME = 16


class SamParseError(ValueError):
    """The response had malformed records; nothing is returned rather than a silent partial result."""


def _resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    rows = (np.arange(height) * mask.shape[0] / height).astype(int)
    cols = (np.arange(width) * mask.shape[1] / width).astype(int)
    return mask[rows][:, cols]


def parse(text: str, index: ClipIndex, concept: str) -> list[Track2D]:
    """Parse one streamed response (all of its text) into one Track2D per SAM object id."""
    parser = video_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        # ASSUMPTION A-113: any malformed record fails the whole response (no silent partial tracks)
        lines = "; ".join(f"line {d.line}: {d.code}" for d in errors[:5])
        raise SamParseError(f"{len(errors)} malformed SAM record(s): {lines}")

    boxes: dict[tuple[int, str], SegmentationBoxRecord] = {}
    masks: dict[tuple[int, str], SegmentationMaskRecord] = {}
    sizes: dict[tuple[int, str], tuple[int, int]] = {}
    for rec in result.records:
        if isinstance(rec, SegmentationBoxRecord) and rec.frame is not None:
            boxes[(rec.frame.frame_index, rec.object_id)] = rec
        elif isinstance(rec, SegmentationMaskRecord) and rec.frame is not None:
            key = (rec.frame.frame_index, rec.object_id)
            if key not in masks or rec.revision > masks[key].revision:
                masks[key] = rec
    sizes.update(_box_sizes(text))

    crop = index.crop
    per_object: dict[str, list[Frame2D]] = defaultdict(list)
    per_frame_count: dict[int, int] = defaultdict(int)
    for (frame, obj), box in sorted(boxes.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        per_frame_count[frame] += 1
        src_w, src_h = sizes[(frame, obj)]
        sx, sy = crop.width / src_w, crop.height / src_h
        # Half-open source box to inclusive clip pixels, then to full-image pixels through the crop offset.
        x1 = crop.x0 + box.left * sx
        y1 = crop.y0 + box.top * sy
        x2 = crop.x0 + box.right * sx - 1
        y2 = crop.y0 + box.bottom * sy - 1
        mask_rle = None
        if (frame, obj) in masks:
            m = masks[(frame, obj)]
            coco = decode_mask_to_rle(m.mask)
            local = rle.decode(MaskRLE(size=coco.size, counts=coco.counts))
            src = np.zeros((src_h, src_w), bool)
            left, top = int(m.bounds.left), int(m.bounds.top)
            bh, bw = int(m.bounds.bottom) - top, int(m.bounds.right) - left
            src[top : top + bh, left : left + bw] = _resize_nearest(local, bh, bw)
            full = np.zeros((crop.full_height, crop.full_width), bool)
            full[crop.y0 : crop.y0 + crop.height, crop.x0 : crop.x0 + crop.width] = _resize_nearest(
                src, crop.height, crop.width
            )
            mask_rle = rle.encode(full)
        cf = index.frame(frame)
        conf = box.confidence if box.confidence is not None else masks.get((frame, obj), box).confidence
        per_object[obj].append(
            Frame2D(
                frame_idx=cf.frame_idx,
                ts=cf.ts,
                box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                mask_rle=mask_rle,
                conf=conf,
            )
        )
    # ASSUMPTION A-110: if any frame returns 16 objects, every track from this response is marked truncated
    truncated = any(n >= MAX_OBJECTS_PER_FRAME for n in per_frame_count.values())
    return [
        Track2D(
            scene_id=index.scene_id,
            camera=index.camera,
            concept=concept,
            obj_id=int(obj),
            truncated=truncated,
            frames=frames,
        )
        for obj, frames in sorted(per_object.items(), key=lambda kv: int(kv[0]))
    ]


# The parser's record grammar (its documented regexes): a line is <Nf>[;fields] then records; a record is an
# object id followed by <|name;fields|> tokens, with an optional comma before the next record.
_HEADER = re.compile(r"^<([0-9]+)f(?:;[^>]*)?>(.*)$")
_RECORD_START = re.compile(r",?([0-9]+)(?=<\|)")
_TOKEN = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(?:;([^|]*))?\|>")


def _box_sizes(text: str) -> dict[tuple[int, str], tuple[int, int]]:
    """The w, h of every record's box token (the source media size SAM reports), by (frame, object id).

    meta-sam-parser validates w and h but does not return them, so they are read here with its grammar.
    """
    sizes: dict[tuple[int, str], tuple[int, int]] = {}
    for line in text.splitlines():
        header = _HEADER.match(line.strip())
        if header is None:
            continue
        frame, body, pos = int(header.group(1)), header.group(2), 0
        while pos < len(body):
            start = _RECORD_START.match(body, pos)
            if start is None:
                break
            end = start.end()
            token = _TOKEN.match(body, end)
            while token is not None:
                if token.group(1) == "box":
                    fields = dict(kv.split("=", 1) for kv in (token.group(2) or "").split(";") if "=" in kv)
                    sizes[(frame, start.group(1))] = (int(fields["w"]), int(fields["h"]))
                end = token.end()
                token = _TOKEN.match(body, end)
            pos = end
    return sizes

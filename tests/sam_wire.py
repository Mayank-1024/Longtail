"""Test helpers that write SAM 3.1 output in its wire format, and a fake SAM transport. SYNTHETIC data only.

meta-sam-parser publishes decoders but no encoder, so masks are encoded with its private one-bit encoder. That
is acceptable for generating test data; production code never imports it.
"""

from pathlib import Path

import numpy as np
from meta_sam_parser._mask_codec import _encode_raster

from longtail.vision.clips import ClipIndex
from longtail.vision.sam_client import TransportError


def record(
    obj_id: int, mask: np.ndarray, conf: float | None = None, scale: float = 1.0, mask_first: bool = False
) -> str:
    """One record for a clip-sized boolean mask; scale < 1 reports it at a smaller source size."""
    m = mask
    if scale != 1.0:
        h, w = int(mask.shape[0] * scale), int(mask.shape[1] * scale)
        m = mask[(np.arange(h) / scale).astype(int)][:, (np.arange(w) / scale).astype(int)]
    vs, us = np.nonzero(m)
    x1, y1, x2, y2 = int(us.min()), int(vs.min()), int(us.max()), int(vs.max())
    local = m[y1 : y2 + 1, x1 : x2 + 1]
    bh, bw = local.shape
    payload = _encode_raster(local.astype(np.uint8).ravel().tolist(), bw, bh)
    c = f";c={conf}" if conf is not None else ""
    box = f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={m.shape[1]};h={m.shape[0]}{c}|>"
    msk = f"<|mask;x=0;y=0;data={bh},{bw},{payload}|>"
    return f"{obj_id}{msk}{box}" if mask_first else f"{obj_id}{box}{msk}"


def frame_line(clip_frame: int, records: list[str]) -> str:
    return f"<{clip_frame}f>" + ",".join(records)


class FakeSamTransport:
    """Answers like SAM 3.1 from the synthetic scene's truth (object silhouettes), in the wire format."""

    def __init__(self, scene, objects: dict[str, list[str]], errors: list[int] | None = None) -> None:
        self.scene, self.objects = scene, objects
        self.errors = list(errors or [])  # HTTP statuses to fail with, in order, before answering
        self.uploads: dict[str, Path] = {}
        self.upload_calls = 0
        self.stream_calls = 0

    def upload(self, path: Path) -> str:
        self.upload_calls += 1
        file_id = f"file-{self.upload_calls}"
        self.uploads[file_id] = Path(path)
        return file_id

    def stream(self, *, model: str, file_id: str, concept: str, include_confidence: bool):
        from synthetic import object_mask

        self.stream_calls += 1
        if self.errors:
            raise TransportError(self.errors.pop(0), "fake")
        index = ClipIndex.model_validate_json(
            self.uploads[file_id].with_suffix(".clip_index.json").read_text()
        )
        c = index.crop
        lines = []
        for cf in index.frames:
            records = []
            for k, name in enumerate(self.objects.get(concept, [])):
                full = object_mask(self.scene, name, index.camera, cf.frame_idx)
                if full is None:
                    continue
                part = full[c.y0 : c.y0 + c.height, c.x0 : c.x0 + c.width]
                if part.any():
                    records.append(record(k, part, conf=0.9 if include_confidence else None))
            if records:
                lines.append(frame_line(cf.clip_frame, records))
        text = "\n".join(lines) + ("\n" if lines else "")
        for i in range(0, len(text), 97):  # streamed in arbitrary chunks
            yield text[i : i + 97]

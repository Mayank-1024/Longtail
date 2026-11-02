"""COCO compressed run-length encoding for binary masks (the format of Frame2D.mask_rle, A-20).

Runs are counted in column-major order and start with a run of zeros. The counts string is the COCO
"compressed" form: each count (after the third, as a difference from the count two before) is written
in 5-bit groups, offset by 48 into printable ASCII.
"""

# ASSUMPTION A-59: our own COCO RLE codec instead of pycocotools; checked against pycocotools offline

import numpy as np

from longtail.contracts import MaskRLE


def _counts(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return [0, *runs] if flat[0] else runs


def _to_string(counts: list[int]) -> str:
    out = []
    for i, c in enumerate(counts):
        x = c - counts[i - 2] if i > 2 else c
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = x != -1 if ch & 0x10 else x != 0
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def _from_string(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            ch = ord(s[p]) - 48
            x |= (ch & 0x1F) << (5 * k)
            more = bool(ch & 0x20)
            p += 1
            k += 1
            if not more and ch & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def encode(mask: np.ndarray) -> MaskRLE:
    """Encode an (h, w) boolean mask."""
    h, w = mask.shape
    return MaskRLE(size=(h, w), counts=_to_string(_counts(mask)))


def decode(rle: MaskRLE) -> np.ndarray:
    """Decode to an (h, w) boolean mask."""
    h, w = rle.size
    counts = _from_string(rle.counts)
    if sum(counts) != h * w:
        raise ValueError(f"RLE counts sum to {sum(counts)}, expected {h * w}")
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape((h, w), order="F")

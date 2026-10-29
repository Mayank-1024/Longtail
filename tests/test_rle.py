import numpy as np
import pytest

from longtail.contracts import MaskRLE
from longtail.vision.rle import decode, encode


def test_matches_pycocotools_reference():
    # Reference produced by pycocotools.mask.encode on this mask (checked offline; A-59).
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    mask[0, 0] = True
    # ASSUMPTION A-59: our codec reproduces pycocotools' compressed counts
    assert encode(mask) == MaskRLE(size=(4, 6), counts="0181J0003")
    np.testing.assert_array_equal(decode(MaskRLE(size=(4, 6), counts="0181J0003")), mask)


def test_round_trips_random_and_edge_masks():
    rng = np.random.default_rng(0)
    masks = [np.zeros((3, 4), bool), np.ones((3, 4), bool), np.zeros((480, 640), bool)]
    masks[-1][100:300, 200:500] = True
    masks += [rng.random((rng.integers(1, 50), rng.integers(1, 50))) < p for p in (0.05, 0.5, 0.95)]
    for m in masks:
        np.testing.assert_array_equal(decode(encode(m)), m)


def test_rejects_counts_of_the_wrong_size():
    with pytest.raises(ValueError, match="sum to"):
        decode(MaskRLE(size=(5, 5), counts=encode(np.ones((4, 4), bool)).counts))

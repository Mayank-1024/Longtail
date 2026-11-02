"""Plan C4 second done check: on 20 real SAM 3.1 masks checked by eye, at least 17 get the correct cluster.

Not runnable yet: it needs SAM 3.1 output (task C3) and PandaSet on disk. When both exist:

1. Pick 20 Track2D frames from development scenes only (never held-out), spread over classes, distances
   and cameras, with a fixed seed. Record the (scene, camera, concept, obj_id, frame_idx) list here.
2. Run associate() on each, and render the mask, the kept cluster (coloured) and the rejected in-mask
   points on the camera image into runs/c4-check/ (git-ignored).
3. Judge each by eye: correct when the kept points lie on the masked object and not on the background,
   the ground or a neighbour. Write the verdicts into docs/eval-report.md.
4. Replace the skip below with an assertion over the recorded verdicts (>= 17 of 20).
"""

import pytest

pytestmark = pytest.mark.skip(reason="needs SAM 3.1 output (C3) and PandaSet; see the module docstring")


def test_at_least_17_of_20_real_masks_correct():
    raise AssertionError("placeholder; see the module docstring")

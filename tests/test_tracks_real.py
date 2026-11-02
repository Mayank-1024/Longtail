"""Plan C5 done checks that need real data or C9. Not runnable yet.

1. A parked car on a real scene reads under 0.3 m/s while the ego moves. Needs PandaSet and SAM 3.1 output
   (C3): pick a development scene where the ego drives past a parked car, associate its Track2Ds (C4), run
   track_motion, and check the speed (also the longest run above 0.5 m/s, which is what C6 acts on).
2. The ego corridor overlays correctly in the bird's-eye view export. Needs the C9 export: draw the
   corridor from ego_path.corridor() on the BEV JSON for a turning segment and judge it by eye.
"""

import pytest


@pytest.mark.skip(reason="needs PandaSet and SAM 3.1 output (C3); see the module docstring")
def test_parked_car_on_a_real_scene_reads_still():
    raise AssertionError("placeholder")


@pytest.mark.skip(reason="needs the C9 bird's-eye view export; see the module docstring")
def test_corridor_overlays_on_the_birds_eye_export():
    raise AssertionError("placeholder")

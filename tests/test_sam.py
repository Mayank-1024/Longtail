"""C3 offline: clips, the SAM 3.1 parser, the client with a fake transport, and SAM tracks into verify()."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from sam_wire import FakeSamTransport
from synthetic import ImageScene, SyntheticConfig, object_mask

from longtail.contracts import CandidateWindow, MaskRLE, ScenarioSpec
from longtail.verify.engine import verify
from longtail.vision import rle
from longtail.vision.clips import ClipIndex, build_clip, check_clip
from longtail.vision.sam_client import (
    MIN_BILLED_FRAMES,
    USD_PER_FRAME,
    Ledger,
    ReplayMissError,
    SamClient,
    TransportError,
)
from longtail.vision.sam_parse import SamParseError, parse
from longtail.vision.track_source import RIGHT_HALF_ID_OFFSET, sam_track_source

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def scene(tmp_path):
    return ImageScene(tmp_path / "images", SyntheticConfig(n_frames=12))


def _client(tmp_path, transport, **kw):
    sleeps = []
    client = SamClient(
        transport,
        cache_dir=tmp_path / "cache" / "sam",
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        replay=False,
        sleep=sleeps.append,
        **kw,
    )
    return client, sleeps


# Clips


def test_clip_round_trip(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    frames = scene.frames("front_camera")
    assert not clip.cached and clip.path.is_file() and clip.index.fps == 10.0
    assert [(f.clip_frame, f.frame_idx, f.ts) for f in clip.index.frames] == [
        (k, f.frame_idx, f.ts) for k, f in enumerate(frames)
    ]
    # Decode every frame: the stripe's grey level encodes the frame index, so order and uniqueness hold.
    w, h = scene.image_size("front_camera")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(clip.path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    decoded = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    assert len(decoded) == len(frames)
    levels = [int(np.median(d[2:6, :, :])) for d in decoded]
    assert levels == pytest.approx([ImageScene.stripe_level(f.frame_idx) for f in frames], abs=4)
    again = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    assert again.cached and again.path == clip.path


def test_concurrent_builders_of_one_clip_all_succeed(scene, tmp_path):
    # A-242: prefetch runs one job per concept, so several workers build the same camera clip at once
    from concurrent.futures import ThreadPoolExecutor

    scene.frames("front_camera")  # the fixture writes its images on first use; real scenes have them on disk
    with ThreadPoolExecutor(max_workers=5) as pool:
        clips = list(
            pool.map(lambda _: build_clip(scene, "front_camera", cache_dir=tmp_path / "clips"), range(5))
        )
    assert len({c.index.clip_id for c in clips}) == 1
    check_clip(
        clips[0].path,
        len(clips[0].index.frames),
        clips[0].index.fps,
        (clips[0].index.crop.width, clips[0].index.crop.height),
    )
    assert not list((tmp_path / "clips").glob("*.tmp*"))


def test_clip_check_catches_a_wrong_frame_count(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    with pytest.raises(RuntimeError, match="decoded frames"):
        check_clip(clip.path, len(clip.index.frames) + 1, clip.index.fps, (640, 480))


def test_crop_offsets_map_boxes_back_to_full_image(scene, tmp_path):
    # ASSUMPTION A-111: half-image crops record their offset; boxes and masks map back to full-image pixels
    clip = build_clip(scene, "front_camera", "right", cache_dir=tmp_path / "clips")
    assert (clip.index.crop.x0, clip.index.crop.width, clip.index.crop.full_width) == (320, 320, 640)
    text = (FIXTURES / "sam31_SYNTHETIC.txt").read_text()
    fixture = ClipIndex.model_validate_json((FIXTURES / "sam31_SYNTHETIC.clip_index.json").read_text())
    shifted = fixture.model_copy(
        update={"crop": fixture.crop.model_copy(update={"name": "right", "x0": 64, "full_width": 128})}
    )
    [full0, *_] = parse(text, fixture, "person")
    [crop0, *_] = parse(text, shifted, "person")
    a, b = full0.frames[0].box_xyxy, crop0.frames[0].box_xyxy
    assert (b[0] - a[0], b[2] - a[2], b[1], b[3]) == (64, 64, a[1], a[3])
    mask = rle.decode(crop0.frames[0].mask_rle)
    assert mask.shape == (48, 128) and mask[10:20, 69:79].all() and not mask[:, :64].any()


# Parser (on the SYNTHETIC fixture)


def _fixture():
    text = (FIXTURES / "sam31_SYNTHETIC.txt").read_text()
    index = ClipIndex.model_validate_json((FIXTURES / "sam31_SYNTHETIC.clip_index.json").read_text())
    return text, index


def test_parser_builds_tracks_from_the_synthetic_fixture():
    text, index = _fixture()
    assert text.startswith("SYNTHETIC")
    tracks = {t.obj_id: t for t in parse(text, index, "person")}
    assert sorted(tracks) == list(range(16))
    t0 = tracks[0]
    # <Nf> markers map through the clip index to camera frames and timestamps.
    assert [(f.frame_idx, f.ts) for f in t0.frames] == [(10, 100.0), (11, 100.1), (12, 100.2), (13, 100.3)]
    # Inclusive boxes in image pixels (A-20).
    assert t0.frames[0].box_xyxy == (5.0, 10.0, 14.0, 19.0)
    # Confidence when present, None when not.
    assert t0.frames[0].conf == pytest.approx(0.9) and tracks[1].frames[0].conf is None
    # Masks decode (COCO RLE) to exactly the boxes' rectangles, also when the mask token comes first.
    m = rle.decode(t0.frames[0].mask_rle)
    assert m.shape == (48, 64) and m.sum() == 100 and m[10:20, 5:15].all()
    assert tracks[1].frames[1].box_xyxy == (31.0, 5.0, 40.0, 9.0)
    assert rle.decode(tracks[1].frames[1].mask_rle)[5:10, 31:41].all()
    # A record reported at half resolution (w=32, h=24) scales back to the clip's pixels.
    x1, y1, x2, y2 = t0.frames[3].box_xyxy
    assert (x1, y1) == (8.0, 10.0) and abs(x2 - 17.0) <= 1 and abs(y2 - 19.0) <= 1


def test_parser_truncation_empty_output_and_errors():
    text, index = _fixture()
    # ASSUMPTION A-110: a frame with 16 objects marks every track of the response truncated
    assert all(t.truncated for t in parse(text, index, "person"))
    without_cap = "\n".join(line for line in text.splitlines() if not line.startswith("<2f>"))
    assert not any(t.truncated for t in parse(without_cap, index, "person"))
    # Empty output is zero matches, not an error.
    assert parse("", index, "person") == [] and parse("\n  \n", index, "person") == []
    # ASSUMPTION A-113: a malformed record fails the whole response
    with pytest.raises(SamParseError, match="malformed"):
        parse("<0f>3<|box;x1=1|>\n", index, "person")


# Client (fake transport)


def test_client_retries_429_and_5xx_but_not_400(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    fake = FakeSamTransport(scene, {"person": ["crosser"]}, errors=[429, 503])
    client, sleeps = _client(tmp_path, fake)
    call = client.segment(clip, "person")
    # ASSUMPTION A-115: backoff 1 s doubling, on 429 and 5xx
    assert call.attempts == 3 and sleeps == [1.0, 2.0] and fake.stream_calls == 3
    bad = FakeSamTransport(scene, {"person": ["crosser"]}, errors=[400])
    client, sleeps = _client(tmp_path / "other", bad)
    with pytest.raises(TransportError, match="400"):
        client.segment(clip, "person")
    assert bad.stream_calls == 1 and sleeps == []


def test_client_reuses_file_id_and_logs_frames_and_cost(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    fake = FakeSamTransport(scene, {"person": ["crosser"], "car": ["parked"]})
    client, _ = _client(tmp_path, fake)
    first = client.segment(clip, "person")
    client.segment(clip, "car")
    # A new client on the same cache still knows the file id.
    client2, _ = _client(tmp_path, fake)
    client2.segment(clip, "person")
    assert fake.upload_calls == 1
    # 12 frames is billed at the 50-frame minimum; the repeat is a cache hit at $0.
    assert first.usd == pytest.approx(MIN_BILLED_FRAMES * USD_PER_FRAME) == pytest.approx(0.01)
    entries = Ledger(tmp_path / "ledger.jsonl").entries()
    assert [(e["concept"], e["frames"], e["usd"], e["cache_hit"]) for e in entries] == [
        ("person", 12, pytest.approx(0.01), False),
        ("car", 12, pytest.approx(0.01), False),
        ("person", 12, 0.0, True),
    ]
    assert fake.stream_calls == 2


def test_replay_serves_the_cache_and_fails_loudly_on_a_miss(scene, tmp_path):
    clip = build_clip(scene, "front_camera", cache_dir=tmp_path / "clips")
    live, _ = _client(tmp_path, FakeSamTransport(scene, {"person": ["crosser"]}))
    live.segment(clip, "person")
    replay = SamClient(
        None, cache_dir=tmp_path / "cache" / "sam", ledger=Ledger(tmp_path / "l.jsonl"), replay=True
    )
    assert replay.segment(clip, "person").cache_hit
    with pytest.raises(ReplayMissError, match="no cached SAM response"):
        replay.segment(clip, "bicycle")


# Into verify()


def test_sam_track_source_feeds_verify_end_to_end(tmp_path):
    s = ImageScene(tmp_path / "images", SyntheticConfig())
    client, _ = _client(tmp_path, FakeSamTransport(s, {"person": ["crosser"]}))
    source = sam_track_source(client, clip_cache=tmp_path / "clips")
    # The masks survive clip -> wire format -> parser intact.
    [track] = source(s, "front_camera", "person")
    f5 = next(f for f in track.frames if f.frame_idx == 5)
    truth = object_mask(s, "crosser", "front_camera", 5)
    assert np.array_equal(rle.decode(MaskRLE.model_validate(f5.mask_rle.model_dump())), truth)
    spec = ScenarioSpec(
        query="a pedestrian crossing in front of us while we move",
        actors=[{"id": "a1", "cls": "pedestrian", "sam_prompt": "person"}],
        predicates=[
            {"name": n, "actors": a, "args": {}}
            for n, a in (("exists", ["a1"]), ("crossing_ego_path", ["a1"]), ("ego_moving", []))
        ],
        assumptions=[],
        clarification=None,
    )
    ts = s.lidar_timestamps()
    window = CandidateWindow(
        window_id="w",
        scene_id=s.scene_id,
        t0=float(ts[0]),
        t1=float(ts[-1]),
        cameras=[],
        score=1.0,
        source="uniform",
    )
    [finding] = verify(spec, [window], s, track_source=source)
    assert finding.verified


def test_halves_offset_right_object_ids(tmp_path):
    # ASSUMPTION A-118: the right half's object ids are offset so the halves never collide
    s = ImageScene(tmp_path / "images", SyntheticConfig(n_frames=12))
    client, _ = _client(tmp_path, FakeSamTransport(s, {"car": ["parked"]}))
    tracks = sam_track_source(client, crop="halves", clip_cache=tmp_path / "clips")(s, "front_camera", "car")
    assert {t.obj_id for t in tracks} == {RIGHT_HALF_ID_OFFSET}  # the parked car sits in the right half


# CLI


def test_cli_without_key_and_replay_miss(tmp_path, monkeypatch, capsys):
    from longtail.cli import main
    from longtail.vision import cli as sam_cli

    s = ImageScene(tmp_path / "images", SyntheticConfig(n_frames=12))
    monkeypatch.setattr(sam_cli, "load_scene", lambda scene_id: s)
    monkeypatch.setenv("LONGTAIL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    # ASSUMPTION A-119: no key, no call: a clear message
    assert main(["sam", "synthetic", "front_camera", "person"]) == 2
    assert "MODEL_API_KEY is not set" in capsys.readouterr().err
    assert main(["sam", "synthetic", "front_camera", "person", "--replay"]) == 1
    assert "replay: no cached SAM response" in capsys.readouterr().err
    # After a (fake) live call fills the cache, replay serves it.
    clip = build_clip(s, "front_camera")
    SamClient(FakeSamTransport(s, {"person": ["crosser"]}), replay=False).segment(clip, "person")
    assert main(["sam", "synthetic", "front_camera", "person", "--replay"]) == 0
    out = capsys.readouterr().out
    assert "1 track(s) from 12 frames (cache hit, $0)" in out and "cost: $0.0000" in out


def test_native_fps_refuses_timestamps_that_do_not_increase():
    import numpy as np
    import pytest

    from longtail.vision.clips import native_fps

    assert native_fps(np.array([0.0, 0.1, 0.2])) == 10.0
    with pytest.raises(ValueError, match="do not increase"):
        native_fps(np.array([5.0, 5.0, 5.0, 5.1]))  # scene 054's camera timestamps (A-240)

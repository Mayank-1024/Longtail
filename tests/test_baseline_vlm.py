"""M5 baseline (a): windows, tiles, claim placement, matching, and C8's file format, with a scripted model."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from longtail.data.base import CameraFrame, Scene, SceneMeta
from longtail.eval.baseline_vlm import (
    CELL_W,
    baseline_windows,
    describe_window,
    match_query,
    run_baseline,
    tile,
)
from longtail.llm.baseten import BasetenTransport
from longtail.llm.cache import ResponseCache
from longtail.llm.client import ModelClient
from longtail.llm.ledger import Ledger

CAMERAS = (
    "front_camera",
    "front_left_camera",
    "front_right_camera",
    "left_camera",
    "right_camera",
    "back_camera",
)
W, H = 192, 108  # small stand-in images; boxes scale to the size the scene reports


class CamScene(Scene):
    """Six cameras and LiDAR timestamps at 10 Hz for 8 s, with real (tiny) JPEGs on disk."""

    def __init__(self, root: Path, scene_id: str = "002", seconds: float = 8.0) -> None:
        object.__setattr__(self, "scene_id", scene_id)
        self._ts = 1000.0 + np.arange(0, seconds, 0.1)
        self._frames = {}
        for k, cam in enumerate(CAMERAS):
            d = root / scene_id / cam
            d.mkdir(parents=True, exist_ok=True)
            path = d / "frame.jpg"
            Image.new("RGB", (W, H), (40 * k, 80, 120)).save(path)
            self._frames[cam] = [CameraFrame(i, float(t), path) for i, t in enumerate(self._ts)]

    @property
    def cameras(self):
        return CAMERAS

    @property
    def meta(self):
        return SceneMeta(self.scene_id, "fake", len(self._ts), float(self._ts[0]), None, None, None, None)

    def frames(self, camera):
        return self._frames[camera]

    def lidar_timestamps(self):
        return self._ts

    def image_size(self, camera):
        return (1920, 1080)

    def lidar(self, ts):  # pragma: no cover - the baseline never reads LiDAR
        raise AssertionError("baseline (a) must not read LiDAR")

    def ego_pose(self, ts):  # pragma: no cover
        raise AssertionError

    def intrinsics(self, camera):  # pragma: no cover
        raise AssertionError

    def extrinsics(self, camera, ts):  # pragma: no cover
        raise AssertionError

    def _labels(self, frame_idx):  # pragma: no cover
        raise AssertionError("baseline (a) must not read labels")

    def _semseg(self, frame_idx):  # pragma: no cover
        raise AssertionError


class Chat:
    """Answers describe calls with `describe` and match calls with `match`; records every payload."""

    def __init__(self, describe, match) -> None:
        self.describe, self.match, self.payloads = describe, match, []

    def create(self, **payload):
        self.payloads.append(payload)
        system = payload["messages"][0]["content"]
        text = self.describe(payload) if "camera images" in system else self.match(payload)
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        }


def client_for(tmp_path: Path, chat: Chat) -> ModelClient:
    return ModelClient(
        BasetenTransport("test", chat=chat),
        cache=ResponseCache(tmp_path / "cache"),
        ledger=Ledger(tmp_path / "ledger.jsonl"),
        allow_paid=True,
        replay=False,
        sleep=lambda s: None,
    )


DESCRIBE = json.dumps(
    {
        "description": "a pedestrian crosses ahead",
        "claims": [
            {
                "image": 0,
                "camera": "front_camera",
                "cls": "pedestrian",
                "box": [0.1, 0.2, 0.3, 0.6],
                "behaviour": "walking",
            },
            {"image": 0, "camera": "front_camera", "cls": "dog", "box": [0, 0, 1, 1], "behaviour": ""},
            {"image": 1, "camera": "roof_camera", "cls": "car", "box": [0, 0, 1, 1], "behaviour": ""},
            {"image": 5, "camera": "front_camera", "cls": "car", "box": [0, 0, 1, 1], "behaviour": ""},
        ],
    }
)


def test_windows_are_2s_and_cover_the_scene(tmp_path):
    ws = baseline_windows(CamScene(tmp_path))
    assert [w[0] for w in ws] == ["002/w000", "002/w001", "002/w002", "002/w003"]
    assert ws[0][1] == pytest.approx(1000.0) and ws[-1][2] == pytest.approx(1007.9)


def test_tile_is_3_by_2_cells_with_one_frame_per_camera(tmp_path):
    jpeg, layout = tile(CamScene(tmp_path), 1000.5)
    import io

    im = Image.open(io.BytesIO(jpeg))
    assert im.size == (3 * CELL_W, 2 * round(CELL_W * 1080 / 1920))
    assert [c["camera"] for c in layout] == list(CAMERAS) and layout[0]["frame_idx"] == 5


def test_describe_places_valid_claims_in_full_image_pixels_and_drops_the_rest(tmp_path):
    scene = CamScene(tmp_path)
    chat = Chat(lambda p: DESCRIBE, lambda p: "{}")
    w = describe_window(scene, "002/w000", 1000.0, 1002.0, client_for(tmp_path, chat))
    assert w.description == "a pedestrian crosses ahead" and w.dropped_claims == 3
    [c] = w.claims
    assert c["claim_id"] == "002/w000/c0" and c["camera"] == "front_camera" and c["frame_idx"] == 0
    assert c["box_xyxy"] == pytest.approx([0.1 * 1919, 0.2 * 1079, 0.3 * 1919, 0.6 * 1079])
    images = [p for p in chat.payloads[0]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2  # one tile per second of a 2 s window


def test_match_keeps_only_known_windows_and_their_own_claims(tmp_path):
    scene = CamScene(tmp_path)
    reply = json.dumps(
        {
            "matches": [
                {"window_id": "002/w000", "score": 1.7, "claim_ids": ["002/w000/c0", "002/w001/c0", "zzz"]},
                {"window_id": "999/w000", "score": 0.9, "claim_ids": []},
            ]
        }
    )
    client = client_for(tmp_path, Chat(lambda p: DESCRIBE, lambda p: reply))
    windows = [describe_window(scene, wid, t0, t1, client) for wid, t0, t1 in baseline_windows(scene)]
    matches, usd = match_query("a pedestrian crossing", windows, client)
    assert [m["window_id"] for m in matches] == ["002/w000"]
    assert matches[0]["score"] == 1.0 and matches[0]["claim_ids"] == ["002/w000/c0"]
    assert usd > 0


def test_run_writes_c8s_files_and_never_sends_a_gold_spec(tmp_path):
    scene = CamScene(tmp_path)
    reply = json.dumps({"matches": [{"window_id": "002/w001", "score": 0.6, "claim_ids": ["002/w001/c0"]}]})
    chat = Chat(lambda p: DESCRIBE, lambda p: reply)
    client = client_for(tmp_path, chat)
    out = tmp_path / "baseline_a"
    queries = [
        {"query_id": "ped_crossing", "text": "a pedestrian crossing in front of us"},
        {"query_id": "bus_stopped", "text": "a stopped bus"},
    ]
    run_baseline([scene], queries, client, out)
    lines = (out / "claims.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["claim_id"] == "002/w000/c0"
    rankings = json.loads((out / "rankings.json").read_text())
    assert rankings["format"] == "longtail.baseline_a/v1" and set(rankings["queries"]) == {
        "ped_crossing",
        "bus_stopped",
    }
    assert rankings["queries"]["ped_crossing"]["cost_usd"] > 0
    sent = json.dumps(chat.payloads)
    assert "predicates" not in sent and "gold" not in sent
    # Describing is cached per window: a second run makes no describe calls.
    before = len(chat.payloads)
    run_baseline([scene], queries[:1], client, out)
    assert all("camera images" not in p["messages"][0]["content"] for p in chat.payloads[before:])


def test_output_loads_with_c8s_reader(tmp_path):
    formats = pytest.importorskip("longtail.eval.formats")
    scene = CamScene(tmp_path)
    reply = json.dumps({"matches": [{"window_id": "002/w001", "score": 0.6, "claim_ids": ["002/w001/c0"]}]})
    client = client_for(tmp_path, Chat(lambda p: DESCRIBE, lambda p: reply))
    run_baseline([scene], [{"query_id": "q", "text": "a pedestrian"}], client, tmp_path / "b")
    claims, rankings = formats.load_baseline(tmp_path / "b")
    assert len(claims) == 4 and rankings.queries["q"].matches[0].claim_ids == ["002/w001/c0"]

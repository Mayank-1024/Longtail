"""Ablation baseline (a), "describe then match" (plan 3.4, M5): Spark Open alone, no SAM and no LiDAR.

1. Describe: for each 2 s window, Spark Open sees a 3x2 tile of the six cameras at one keyframe per second and
   returns a description plus box claims {class, camera, box, behaviour}. Cached per window (the M1 cache),
   so every query reuses them.
2. Match: each query's text is matched against the window descriptions; the model returns the windows that
   match, a score and the claims each rests on.
Output: runs/baseline_a/claims.jsonl and rankings.json in C8's format (A-135, docs/eval-formats.md). C8 then
runs the verifier on every cited box, which gives the hallucination-catch rate.

The baseline sees only query text, never a gold spec or a label.
"""

import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from longtail.contracts import TAXONOMY
from longtail.data.base import CameraFrame, Scene
from longtail.llm.baseten import SPARK_OPEN
from longtail.llm.client import ModelClient, ModelResponse, token_usd

FORMAT = "longtail.baseline_a/v1"
OUT_DIR = Path("runs") / "baseline_a"
# ASSUMPTION A-221: 2 s windows with no overlap, one keyframe per second, cells 640 px wide in a 3x2 tile
WINDOW_S = 2.0
KEYFRAME_STEP_S = 1.0
CELL_W = 640
# ASSUMPTION A-222: matching sends at most 40 window descriptions per call; scores are clamped to [0, 1]
MATCH_CHUNK = 40

DESCRIBE_SYSTEM = f"""You look at driving-log camera images: each image is a 3x2 tile of six cameras at one
moment, every cell labelled with its camera name. Describe the scene, then list the objects you can see.

Return only JSON: {{"description": str, "claims": [{{"image": int, "camera": str, "cls": str,
"box": [x1, y1, x2, y2], "behaviour": str}}]}}
- image is the 0-based index of the tile image; camera is the label of the cell the object is in.
- cls is one of: {", ".join(TAXONOMY)}. Skip anything else.
- box is in that cell's own coordinates, as fractions from 0 to 1 (x right, y down).
- behaviour is a few words (e.g. "walking across", "parked", "turning left"), or "" if unclear."""

MATCH_SYSTEM = """You search driving-log windows by their text descriptions. Given a query and a list of
windows (each with a description and its claimed objects), return the windows that show what the query asks.

Return only JSON: {"matches": [{"window_id": str, "score": float, "claim_ids": [str]}]}
- score from 0 to 1: how well the window matches.
- claim_ids: the claims in that window that make it match, one per object the query names.
- Leave out windows that do not match."""


@dataclass
class WindowDescription:
    window_id: str
    scene_id: str
    t0: float
    t1: float
    description: str
    claims: list[dict[str, Any]]
    list_usd: float
    dropped_claims: int = 0


@dataclass
class BaselineRun:
    windows: list[WindowDescription] = field(default_factory=list)
    rankings: dict[str, Any] = field(default_factory=dict)


def baseline_windows(scene: Scene) -> list[tuple[str, float, float]]:
    ts = scene.lidar_timestamps()
    t_first, t_last = float(ts[0]), float(ts[-1])
    out = []
    t0, i = t_first, 0
    while t0 < t_last - 1e-6 or i == 0:
        out.append((f"{scene.scene_id}/w{i:03d}", t0, min(t0 + WINDOW_S, t_last)))
        t0, i = t0 + WINDOW_S, i + 1
    return out


def _nearest(frames: list[CameraFrame], t: float) -> CameraFrame:
    times = np.array([f.ts for f in frames])
    return frames[int(np.argmin(np.abs(times - t)))]


def tile(scene: Scene, t: float, cell_w: int = CELL_W) -> tuple[bytes, list[dict[str, Any]]]:
    """A 3x2 JPEG of the six cameras at the frames nearest t, and where each camera's cell is."""
    cams = list(scene.cameras)[:6]
    w0, h0 = scene.image_size(cams[0])
    cell_h = round(cell_w * h0 / w0)
    canvas = Image.new("RGB", (3 * cell_w, 2 * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    layout = []
    for k, cam in enumerate(cams):
        frame = _nearest(scene.frames(cam), t)
        x, y = (k % 3) * cell_w, (k // 3) * cell_h
        with Image.open(frame.image_path) as im:
            canvas.paste(im.convert("RGB").resize((cell_w, cell_h)), (x, y))
        draw.rectangle([x, y, x + 8 * len(cam) + 8, y + 16], fill=(0, 0, 0))
        draw.text((x + 4, y + 2), cam, fill=(255, 255, 0))
        layout.append({"camera": cam, "frame_idx": frame.frame_idx, "ts": frame.ts})
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), layout


def _json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    return json.loads(body[start : end + 1] if start >= 0 and end > start else body)


def _list_usd(resp: ModelResponse, model: str) -> float:
    # C8 costs at list price with cache hits included (its A-13), so a hit is priced by its usage
    return token_usd(model, resp.usage.get("input_tokens", 0), resp.usage.get("output_tokens", 0))


def describe_window(
    scene: Scene, window_id: str, t0: float, t1: float, client: ModelClient, model: str = SPARK_OPEN
) -> WindowDescription:
    images, layouts = [], []
    t = t0
    while t < t1 - 1e-6 or not images:
        jpeg, layout = tile(scene, t)
        images.append(jpeg)
        layouts.append(layout)
        t += KEYFRAME_STEP_S
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": f"{len(images)} tile image(s), one per second."}
    ]
    content += [
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}
        for j in images
    ]
    request = {
        "model": model,
        "input": [{"role": "system", "content": DESCRIBE_SYSTEM}, {"role": "user", "content": content}],
        "max_output_tokens": 1200,
        "temperature": 0,
    }
    resp = client.respond(request, purpose="baseline_describe")
    try:
        data = _json(resp.text)
    except (ValueError, json.JSONDecodeError):
        data = {}
    claims, dropped = [], 0
    for c in data.get("claims") or [] if isinstance(data, dict) else []:
        claim = _to_claim(scene, window_id, len(claims), c, layouts)
        if claim is None:
            dropped += 1
        else:
            claims.append(claim)
    desc = str(data.get("description", "")) if isinstance(data, dict) else ""
    return WindowDescription(window_id, scene.scene_id, t0, t1, desc, claims, _list_usd(resp, model), dropped)


def _to_claim(
    scene: Scene, window_id: str, n: int, c: Any, layouts: list[list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """A model claim to a C8 VlmClaim dict, with the box in full-image pixels; None if it cannot be placed."""
    if not isinstance(c, dict) or c.get("cls") not in TAXONOMY:
        return None
    image = c.get("image", 0)
    if not isinstance(image, int) or not 0 <= image < len(layouts):
        return None
    cell = next((x for x in layouts[image] if x["camera"] == c.get("camera")), None)
    box = c.get("box")
    if cell is None or not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    w, h = scene.image_size(cell["camera"])
    return {
        "claim_id": f"{window_id}/c{n}",
        "scene_id": scene.scene_id,
        "camera": cell["camera"],
        "frame_idx": int(cell["frame_idx"]),
        "ts": float(cell["ts"]),
        "cls": c["cls"],
        "box_xyxy": [x1 * (w - 1), y1 * (h - 1), x2 * (w - 1), y2 * (h - 1)],
        "behaviour": str(c.get("behaviour") or "") or None,
    }


def match_query(
    text: str, windows: list[WindowDescription], client: ModelClient, model: str = SPARK_OPEN
) -> tuple[list[dict[str, Any]], float]:
    """The windows matching a query, each with a score and its claims, and the list price of the calls."""
    by_id = {w.window_id: w for w in windows}
    matches: dict[str, dict[str, Any]] = {}
    usd = 0.0
    for i in range(0, len(windows), MATCH_CHUNK):
        chunk = [
            {
                "window_id": w.window_id,
                "description": w.description,
                "claims": [
                    {
                        "claim_id": c["claim_id"],
                        "cls": c["cls"],
                        "camera": c["camera"],
                        "behaviour": c["behaviour"],
                    }
                    for c in w.claims
                ],
            }
            for w in windows[i : i + MATCH_CHUNK]
        ]
        request = {
            "model": model,
            "input": [
                {"role": "system", "content": MATCH_SYSTEM},
                {"role": "user", "content": json.dumps({"query": text, "windows": chunk})},
            ],
            "max_output_tokens": 1500,
            "temperature": 0,
        }
        resp = client.respond(request, purpose="baseline_match")
        usd += _list_usd(resp, model)
        try:
            data = _json(resp.text)
        except (ValueError, json.JSONDecodeError):
            continue
        for m in data.get("matches") or [] if isinstance(data, dict) else []:
            w = by_id.get(m.get("window_id")) if isinstance(m, dict) else None
            if w is None:
                continue
            own = {c["claim_id"] for c in w.claims}
            claim_ids = [c for c in m.get("claim_ids") or [] if c in own]
            try:
                score = min(1.0, max(0.0, float(m.get("score", 0.0))))
            except (TypeError, ValueError):
                score = 0.0
            prev = matches.get(w.window_id)
            if prev is None or score > prev["score"]:
                matches[w.window_id] = {
                    "window_id": w.window_id,
                    "scene_id": w.scene_id,
                    "t0": w.t0,
                    "t1": w.t1,
                    "score": score,
                    "claim_ids": claim_ids,
                }
    return sorted(matches.values(), key=lambda m: -m["score"]), usd


DESCRIBE_WORKERS = 8


def run_baseline(
    scenes: list[Scene],
    queries: list[dict[str, str]],
    client: ModelClient,
    out_dir: Path = OUT_DIR,
    model: str = SPARK_OPEN,
) -> BaselineRun:
    """Describe every window of every scene once, match every query, write C8's two files.

    queries: [{"query_id": ..., "text": ...}]; only the text is sent to the model.
    """
    start = time.perf_counter()
    jobs = [(s, wid, t0, t1) for s in scenes for wid, t0, t1 in baseline_windows(s)]
    # ASSUMPTION A-235: windows are described by up to 8 threads (network-bound calls); results keep job order
    with ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        windows = list(pool.map(lambda j: describe_window(*j, client, model), jobs))
    describe_s = time.perf_counter() - start
    describe_usd = sum(w.list_usd for w in windows)
    share = 1 / max(1, len(queries))
    rankings: dict[str, Any] = {"format": FORMAT, "model": model, "queries": {}}
    for q in queries:
        t = time.perf_counter()
        matches, usd = match_query(q["text"], windows, client, model)
        # ASSUMPTION A-223: describing is shared by all queries, so each query carries an equal share of its
        # cost and time on top of its own matching calls
        rankings["queries"][q["query_id"]] = {
            "matches": matches,
            "cost_usd": round(usd + describe_usd * share, 6),
            "seconds": round(time.perf_counter() - t + describe_s * share, 3),
        }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "claims.jsonl", "w") as f:
        for w in windows:
            for c in w.claims:
                f.write(json.dumps(c) + "\n")
    (out / "rankings.json").write_text(json.dumps(rankings, indent=1))
    return BaselineRun(windows, rankings)

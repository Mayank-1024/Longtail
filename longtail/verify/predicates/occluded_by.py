"""occluded_by(a, cls): a nearer cluster of class cls covers enough of the target's amodal box (plan C6)."""

from typing import Any

import numpy as np

from longtail.geometry.project import project
from longtail.geometry.tracks3d import TrackedObject
from longtail.geometry.transforms import world_to_camera
from longtail.verify.predicates.base import Context, FrameResult, Reason, missing, sample_at


def threshold(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> str:
    c = ctx.cfg.predicates.occluded_by
    return f"a nearer {args['cls']} covers >= {c.min_cover_frac:.0%} of the target's amodal box"


def _pixel(k: np.ndarray, world_from_camera: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    p = world_to_camera(world_from_camera, point[None])[0]
    if p[2] <= 0:
        return None
    return np.array([k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]])


def _amodal_box(
    ctx: Context, target: TrackedObject, cluster, pose, k, size
) -> tuple[np.ndarray, float] | None:
    """(u0, v0, u1, v1) and median depth of the target in one camera, extended to the ground and up to the
    class's typical height (A-07)."""
    pts = ctx.scene.lidar(cluster.ts).points[cluster.point_index, :3]
    proj = project(pts, pose, k, size)
    if len(proj) == 0:
        return None
    u0, v0 = proj.uv.min(axis=0)
    u1, v1 = proj.uv.max(axis=0)
    centre = np.median(pts, axis=0)
    plane = cluster.plane
    if plane is not None and abs(plane.normal[2]) > 1e-9:
        ground_z = -(plane.normal[0] * centre[0] + plane.normal[1] * centre[1] + plane.d) / plane.normal[2]
    else:
        ground_z = float(pts[:, 2].min())
    # ASSUMPTION A-07: the visible box extended down to the local ground and up to the class-prior height
    typical = ctx.cfg.predicates.exists.size_priors[target.track.cls].typical_height_m
    for z, extend in ((ground_z, "down"), (ground_z + typical, "up")):
        px = _pixel(k, pose, np.array([centre[0], centre[1], z]))
        if px is not None:
            v0, v1 = (min(v0, px[1]), v1) if extend == "up" else (v0, max(v1, px[1]))
    # ASSUMPTION A-93: also the C5 box (track-wide size, so a part hidden at the side counts)
    fit = target.box
    j = fit.at(cluster.ts) if fit is not None else None
    if j is not None and not np.isnan(fit.centres[j]).any():
        c, s = np.cos(fit.yaws[j]), np.sin(fit.yaws[j])
        axes = np.array([[c, s], [-s, c]]) * (fit.size[:, None] / 2)
        corners = [
            np.array([*(fit.centres[j, :2] + sx * axes[0] + sy * axes[1]), z])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for z in (ground_z, ground_z + typical)
        ]
        px = [_pixel(k, pose, p) for p in corners]
        if all(p is not None for p in px):
            px = np.array(px)
            u0, v0 = min(u0, px[:, 0].min()), min(v0, px[:, 1].min())
            u1, v1 = max(u1, px[:, 0].max()), max(v1, px[:, 1].max())
    w, h = size
    box = np.array([max(u0, 0), max(v0, 0), min(u1, w - 1), min(v1, h - 1)])
    return box, float(np.median(proj.depth))


def evaluate(ctx: Context, objs: list[TrackedObject], args: dict[str, Any]) -> list[FrameResult]:
    [target] = objs
    c = ctx.cfg.predicates.occluded_by
    occluders = [o for o in ctx.pool(args["cls"]) if o.track.track_id != target.track.track_id]
    out = []
    for ts in ctx.frames:
        if sample_at(target, ts) is None:
            miss = missing(ctx, target, ts)
            # Plan C6: UNVERIFIABLE when the target has no LiDAR, whatever the reason.
            out.append(
                miss if miss.status == "UNVERIFIABLE" else FrameResult.unverifiable(Reason.NO_COVERAGE)
            )
            continue
        best, best_detail, seen = 0.0, {}, False
        for cl in target.association.clusters[ts]:
            frame = ctx.scene.frames(cl.camera)[cl.frame_idx]
            pose = ctx.scene.extrinsics(cl.camera, frame.ts)
            k, size = ctx.scene.intrinsics(cl.camera), ctx.scene.image_size(cl.camera)
            amodal = _amodal_box(ctx, target, cl, pose, k, size)
            if amodal is None:
                continue
            seen = True
            box, depth = amodal
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
            for o in occluders:
                for oc in o.association.clusters.get(ts, []):
                    pts = ctx.scene.lidar(ts).points[oc.point_index, :3]
                    p = project(pts, pose, k, size)
                    if len(p) == 0 or np.median(p.depth) >= depth:
                        continue
                    # ASSUMPTION A-84: coverage = occluder's projected bounding box over the amodal box
                    ou0, ov0 = p.uv.min(axis=0)
                    ou1, ov1 = p.uv.max(axis=0)
                    iw = max(0.0, min(box[2], ou1) - max(box[0], ou0))
                    ih = max(0.0, min(box[3], ov1) - max(box[1], ov0))
                    cover = iw * ih / area
                    if cover > best:
                        best, best_detail = cover, {"camera": cl.camera, "occluder": o.track.track_id}
        if not seen:
            out.append(FrameResult.unverifiable(Reason.NO_COVERAGE))
            continue
        ok = best >= c.min_cover_frac
        detail = {"covered_frac": round(best, 3), **best_detail}
        out.append(
            FrameResult(
                "PASS" if ok else "FAIL",
                Reason.OK if ok else Reason.NOT_OCCLUDED,
                best,
                best - c.min_cover_frac,
                detail,
            )
        )
    return out

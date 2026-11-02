import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { Bev, FindingDetail } from "../types";
import { BevView } from "./Bev";
import { CardView, StatusBadge } from "./CardView";

const fmt = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

export function Detail({ sid, fid, onBack, onChanged }: {
  sid: string;
  fid: string;
  onBack: () => void;
  onChanged: () => void;
}) {
  const [detail, setDetail] = useState<FindingDetail | null>(null);
  const [bev, setBev] = useState<Bev | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = () =>
    api.finding(sid, fid).then((d) => {
      setDetail(d);
      if (d.viz) api.bev(d.viz.bev_url).then(setBev).catch((e) => setError(String(e)));
    });

  useEffect(() => {
    setDetail(null);
    setBev(null);
    setPos(0);
    load().catch((e) => setError(String(e)));
  }, [sid, fid]);

  const frames = useMemo(
    () => (detail?.viz?.timeline.frames ?? []).filter((f) => f.in_interval).map((f) => f.frame_idx),
    [detail],
  );
  const frameIdx = frames[Math.min(pos, Math.max(0, frames.length - 1))];

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const t = setInterval(() => setPos((p) => (p + 1) % frames.length), 100); // 10 Hz, the sweep rate
    return () => clearInterval(t);
  }, [playing, frames.length]);

  if (error) return <div className="error">{error}</div>;
  if (!detail) return <div className="loading">exporting overlays…</div>;
  const { finding, spec, card, viz } = detail;
  const overlays = (viz?.overlays ?? []).filter((o) => o.frame_idx === frameIdx);
  const missing = (viz?.missing_overlays ?? []).filter((o) => o.frame_idx === frameIdx);
  const bevFrame = bev?.frames.find((f) => f.frame_idx === frameIdx);
  const t0 = viz?.timeline.frames[0]?.ts ?? finding.t0;

  const decide = async (decision: "approve" | "reject") => {
    setBusy(true);
    try {
      await api.review(sid, fid, decision);
      await load();
      onChanged();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const recard = async () => {
    setBusy(true);
    try {
      await api.writeCard(sid, fid);
      await load();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="detail">
      <div className="detail-head">
        <button onClick={onBack}>← results</button>
        <span className={finding.verified ? "verdict verdict-ok" : "verdict verdict-no"}>
          {finding.verified ? "VERIFIED" : "REJECTED BY VERIFIER"}
        </span>
        <span className="muted">
          scene {finding.scene_id} · {fmt(finding.t0 - t0, 1)}–{fmt(finding.t1 - t0, 1)} s · score {fmt(finding.score)}
        </span>
        <span className="spacer" />
        <button disabled={busy} onClick={recard}>write card</button>
        <button disabled={busy || !finding.verified} onClick={() => decide("approve")}>approve</button>
        <button disabled={busy} onClick={() => decide("reject")}>reject</button>
        {detail.review && <span className={`review review-${detail.review.decision}`}>{detail.review.decision}d</span>}
      </div>

      <CardView card={card} results={finding.results} />

      {viz ? (
        <>
          <div className="scrubber">
            <button onClick={() => setPlaying((p) => !p)}>{playing ? "pause" : "play"}</button>
            <input type="range" min={0} max={Math.max(0, frames.length - 1)} value={pos}
              onChange={(e) => { setPlaying(false); setPos(Number(e.target.value)); }} />
            <span className="muted">sweep {frameIdx} · {pos + 1}/{frames.length}</span>
          </div>
          <div className="views">
            <div className="cameras">
              {overlays.map((o) => (
                <figure key={o.camera}>
                  <img src={o.url} alt={`${o.camera} sweep ${o.frame_idx}`} loading="lazy" />
                  <figcaption>{o.camera.replace("_camera", "")}</figcaption>
                </figure>
              ))}
              {missing.map((m) => (
                <figure key={m.camera} className="missing">
                  <div>no synced frame</div>
                  <figcaption>{m.camera.replace("_camera", "")}</figcaption>
                </figure>
              ))}
            </div>
            <div className="bev-wrap">
              <BevView frame={bevFrame} actorOrder={spec.actors.map((a) => a.id)} />
              <div className="muted small">bird's-eye view, ego frame: forward is up; rings every 10 m</div>
            </div>
          </div>
        </>
      ) : (
        <div className="muted">no visual export for this finding</div>
      )}

      <table className="predicates">
        <thead>
          <tr>
            <th>predicate</th>
            <th>actors</th>
            <th>status</th>
            <th>value</th>
            <th>threshold</th>
            <th>reason</th>
            <th>per sweep</th>
          </tr>
        </thead>
        <tbody>
          {finding.results.map((r) => {
            const strip = viz?.timeline.predicates.find((p) => p.result_id === r.result_id);
            return (
              <tr key={r.result_id} className={r.required ? "" : "optional"}>
                <td>{r.name}</td>
                <td>{Object.entries(r.actors).map(([a, t]) => `${a}=${t}`).join(" ") || "ego"}</td>
                <td><StatusBadge status={r.status} /></td>
                <td>{fmt(r.value)}</td>
                <td>{r.threshold}</td>
                <td>{r.evidence.reason}</td>
                <td>
                  <div className="strip">
                    {strip?.entries.map((e) => (
                      <span key={e.frame_idx}
                        className={`cell cell-${e.status.toLowerCase()}${e.frame_idx === frameIdx ? " cell-now" : ""}`}
                        title={`sweep ${e.frame_idx}: ${e.status} ${e.reason ?? ""}`}
                        onClick={() => {
                          const i = frames.indexOf(e.frame_idx);
                          if (i >= 0) { setPlaying(false); setPos(i); }
                        }} />
                    ))}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {spec.assumptions.length > 0 && (
        <div className="assumptions">assumptions: {spec.assumptions.join(" · ")}</div>
      )}
    </div>
  );
}

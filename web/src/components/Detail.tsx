import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type { Bev, FindingDetail } from "../types";
import { BevView } from "./Bev";
import { CardView, StatusBadge } from "./CardView";
import { StepLoader } from "./Loader";

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
  const [busy, setBusy] = useState<"card" | "approve" | "reject" | null>(null);
  const [loaded, setLoaded] = useState<Set<string>>(new Set()); // camera images that have arrived
  const [ready, setReady] = useState(false); // the first frame's images and the bird's-eye view are in
  const [big, setBig] = useState<string | null>(null); // the camera shown enlarged
  const [toast, setToast] = useState<string | null>(null);
  const keys = useRef<(e: KeyboardEvent) => void>(() => undefined);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA") && (t as HTMLInputElement).type !== "range") return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      keys.current(e);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(t);
  }, [toast]);

  const load = () =>
    api.finding(sid, fid).then((d) => {
      setDetail(d);
      if (d.viz) api.bev(d.viz.bev_url).then(setBev).catch((e) => setError(String(e)));
    });

  useEffect(() => {
    setDetail(null);
    setBev(null);
    setPos(0);
    setLoaded(new Set());
    setReady(false);
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

  const firstOverlays = (detail?.viz?.overlays ?? []).filter((o) => o.frame_idx === frames[0]);
  const imgsIn = firstOverlays.filter((o) => loaded.has(o.url)).length;
  const bevIn = bev !== null;
  useEffect(() => {
    if (detail && !ready && (!detail.viz || (bevIn && imgsIn >= firstOverlays.length))) setReady(true);
  }, [detail, ready, bevIn, imgsIn, firstOverlays.length]);
  useEffect(() => {
    if (!detail || ready) return;
    const t = setTimeout(() => setReady(true), 15000); // never leave the page dimmed if an image never reports
    return () => clearTimeout(t);
  }, [detail, ready]);
  const arrived = (url: string) => setLoaded((prev) => (prev.has(url) ? prev : new Set(prev).add(url)));

  if (error) return <div className="error">{error}</div>;
  if (!detail)
    return (
      <StepLoader title="Opening the finding" sub="Reading the verifier's results and preparing the evidence."
        stages={["Results + overlays", "Bird's-eye view", "Camera frames"]} stage={0}
        why={"The first time a finding in a scene is opened, the server draws the tracked boxes onto every " +
          "camera frame and builds the LiDAR bird's-eye view for all of that scene's findings, then reuses " +
          "them. This takes a few seconds once; after that, findings open at once."} />
    );
  const { finding, spec, card, viz } = detail;
  const overlays = (viz?.overlays ?? []).filter((o) => o.frame_idx === frameIdx);
  const missing = (viz?.missing_overlays ?? []).filter((o) => o.frame_idx === frameIdx);
  const bevFrame = bev?.frames.find((f) => f.frame_idx === frameIdx);
  const t0 = viz?.timeline.frames[0]?.ts ?? finding.t0;

  const decide = async (decision: "approve" | "reject") => {
    setBusy(decision);
    try {
      await api.review(sid, fid, decision);
      setToast(decision === "approve" ? "Approved: this finding is now in the JSON and CSV export." : "Rejected: kept out of the export.");
      await load();
      onChanged();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };
  const recard = async () => {
    setBusy("card");
    try {
      await api.writeCard(sid, fid);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(null);
    }
  };

  keys.current = (e: KeyboardEvent) => {
    if (e.key === "Escape") { if (big) setBig(null); else onBack(); }
    else if (e.key === " " && frames.length > 1) { e.preventDefault(); setPlaying((p) => !p); }
    else if (e.key === "ArrowRight" && frames.length) { e.preventDefault(); setPlaying(false); setPos((p) => Math.min(frames.length - 1, p + 1)); }
    else if (e.key === "ArrowLeft" && frames.length) { e.preventDefault(); setPlaying(false); setPos((p) => Math.max(0, p - 1)); }
    else if (e.key === "a" && busy === null && finding.verified) decide("approve");
    else if (e.key === "r" && busy === null) decide("reject");
  };

  return (
    <div className="detail fade-in">
      {toast && <div className="toast" role="status">{toast}</div>}
      <div className="detail-head">
        <button className="back" onClick={onBack}>← results</button>
        <span className={finding.verified ? "verdict verdict-ok" : "verdict verdict-no"}>
          {finding.verified ? "VERIFIED" : "REJECTED BY VERIFIER"}
        </span>
        <span className="muted">
          scene {finding.scene_id} · {fmt(finding.t0 - t0, 1)}–{fmt(finding.t1 - t0, 1)} s · score {fmt(finding.score)}
        </span>
        <span className="spacer" />
        <button disabled={busy !== null} onClick={recard}>{busy === "card" ? "writing…" : "write card"}</button>
        <button className="btn-approve" disabled={busy !== null || !finding.verified} onClick={() => decide("approve")}>
          {busy === "approve" ? "saving…" : "approve"}
        </button>
        <button className="btn-reject" disabled={busy !== null} onClick={() => decide("reject")}>
          {busy === "reject" ? "saving…" : "reject"}
        </button>
        {detail.review && <span className={`review review-${detail.review.decision}`}>{detail.review.decision}d</span>}
      </div>

      {busy === "card" ? (
        <StepLoader title="Muse Spark Open is writing the evidence card"
          sub="It sees only the verifier's measurements; every sentence must cite one."
          stages={["Request sent", "Writing + checking citations", "Show card"]} stage={1}
          why="The validator rejects any sentence that cites no measurement or states a failed check as fact, so a card can take a few seconds." />
      ) : (
        <CardView card={card} results={finding.results} />
      )}

      {viz && !ready && (
        <StepLoader compact title={`Loading the evidence · ${imgsIn} of ${firstOverlays.length} camera frames`}
          sub={bevIn ? "Bird's-eye view ready." : "Loading the LiDAR bird's-eye view…"}
          done={imgsIn} total={firstOverlays.length} unit="frames"
          stages={["Results + overlays", "Bird's-eye view", "Camera frames"]} stage={bevIn ? 2 : 1} delayMs={150} />
      )}

      {viz ? (
        <>
          <div className="scrubber">
            <button className="play" onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ pause" : "▶ play"}</button>
            <input type="range" min={0} max={Math.max(0, frames.length - 1)} value={pos}
              onChange={(e) => { setPlaying(false); setPos(Number(e.target.value)); }} />
            <span className="muted">sweep {frameIdx} · {pos + 1}/{frames.length}</span>
          </div>
          <div className="kbd-hint muted small">
            <kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> step · <kbd>a</kbd> approve · <kbd>r</kbd> reject ·
            <kbd>esc</kbd> back · click a camera to enlarge
          </div>
          <div className="views">
            <div className="cameras">
              {overlays.map((o) => (
                <figure key={o.camera} className={big === o.camera ? "big" : ""}
                  onClick={() => setBig((b) => (b === o.camera ? null : o.camera))}
                  title={big === o.camera ? "click to shrink" : "click to enlarge"}>
                  <img src={o.url} alt={`${o.camera} sweep ${o.frame_idx}`} loading="lazy"
                    className={ready || loaded.has(o.url) ? "img-in" : "img-wait"}
                    onLoad={() => arrived(o.url)} onError={() => arrived(o.url)} />
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
              {bev ? (
                <BevView frame={bevFrame} actorOrder={spec.actors.map((a) => a.id)} />
              ) : (
                <div className="bev-loading">loading the LiDAR bird's-eye view…</div>
              )}
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

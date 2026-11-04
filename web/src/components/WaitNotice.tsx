import { useEffect, useRef, useState } from "react";
import type { TraceEvent } from "../types";

// 5 min a scene: measured on a laptop with 4 SAM workers (about 2.5 h over the 29 index scenes). Once a
// scene has finished, the estimate uses the measured pace of this search instead.
const MIN_PER_SCENE = 5;

type SceneState = "pending" | "found" | "none" | "error";

function clock(s: number): string {
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m}:${String(s % 60).padStart(2, "0")}`;
}

function eta(sec: number): string {
  if (sec < 60) return "under a minute left";
  const m = Math.round(sec / 60);
  return m >= 90 ? `about ${(m / 60).toFixed(1)} h left` : `about ${m} min left`;
}

const STAGES = ["Plan", "Pick scenes", "Track (SAM 3.1)", "Verify (LiDAR)", "Finish"] as const;

function Ring({ done, total, spinning }: { done: number; total: number; spinning: boolean }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  const frac = total > 0 ? done / total : 0;
  return (
    <svg className="ring" viewBox="0 0 84 84" width="84" height="84" aria-hidden="true">
      <circle cx="42" cy="42" r={r} className="ring-track" />
      {spinning ? (
        <circle cx="42" cy="42" r={r} className="ring-spin" strokeDasharray={`${c * 0.22} ${c}`} />
      ) : (
        <circle cx="42" cy="42" r={r} className="ring-fill" strokeDasharray={`${c * frac} ${c}`}
          transform="rotate(-90 42 42)" />
      )}
      <text x="42" y="40" className="ring-num">{total > 0 ? `${done}/${total}` : "…"}</text>
      <text x="42" y="55" className="ring-label">{total > 0 ? "scenes" : ""}</text>
    </svg>
  );
}

/** While a search runs: an animated loader with the stage, scenes verified and left, an estimate, and why. */
export function WaitNotice({ events, startedAt }: { events: TraceEvent[]; startedAt: number }) {
  const [now, setNow] = useState(Date.now());
  const seen = useRef<number[]>([]); // when each event arrived, for the measured pace
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  while (seen.current.length < events.length) seen.current.push(Date.now());

  const elapsed = Math.max(0, Math.round((now - startedAt) / 1000));
  const order: string[] = [];
  let lastSearching = -1;
  let lastTracked = -1;
  events.forEach((e, i) => {
    if (e.name === "searching") {
      lastSearching = i;
      for (const s of (e.detail.scene_ids as string[] | undefined) ?? []) order.push(s);
    }
    if (e.name === "tracked") lastTracked = i;
  });
  const state: Record<string, SceneState> = Object.fromEntries(order.map((s) => [s, "pending" as SceneState]));
  let verified = 0;
  const doneAt: number[] = [];
  events.forEach((e, i) => {
    if (e.name === "scene_done") {
      const d = e.detail as { scene_id: string; verified?: number; error?: string };
      state[d.scene_id] = d.error ? "error" : (d.verified ?? 0) > 0 ? "found" : "none";
      verified += d.verified ?? 0;
      doneAt.push(seen.current[i]);
    }
  });
  const total = order.length;
  const done = doneAt.length;
  const left = total - done;
  const tracking = total > 0 && lastTracked < lastSearching;
  const last = events[events.length - 1];

  let stage = 0;
  if (events.some((e) => e.kind === "plan")) stage = 1;
  if (total > 0) stage = tracking ? 2 : 3;
  if (events.some((e) => e.name === "finish") || (total > 0 && left === 0 && last?.kind === "model")) stage = 4;

  let headline: string;
  let sub: string;
  if (stage === 0) {
    headline = "Planning the search";
    sub = "Muse Wonder is turning your sentence into a scenario spec: actors, relations, durations.";
  } else if (stage === 1) {
    headline = "Picking the scenes to search";
    sub = "The agent reads the scene index and chooses where this scenario is likely to happen.";
  } else if (stage === 2) {
    headline = `Tracking objects in ${total} scene${total === 1 ? "" : "s"}`;
    sub = "SAM 3.1 segments and tracks every named object through all six cameras.";
  } else if (stage === 3) {
    const since = lastTracked >= 0 ? seen.current[lastTracked] : startedAt;
    const pace = done > 0 ? (doneAt[done - 1] - since) / 1000 / done : MIN_PER_SCENE * 60;
    headline = `${done} of ${total} scene${total === 1 ? "" : "s"} verified · ${left} left`;
    sub = `Checking every track against the LiDAR, frame by frame · ${eta(pace * left)}${done > 0 ? "" : " (estimate)"}`;
  } else {
    headline = "Finishing";
    sub = "The agent is collecting the verified findings.";
  }

  return (
    <div className="loader" role="status" aria-live="polite">
      <div className="loader-top">
        <Ring done={done} total={total} spinning={stage < 3} />
        <div className="loader-text">
          <div className="loader-head">{headline}</div>
          <div className="loader-sub">{sub}</div>
          <div className="loader-meta">
            <span>elapsed {clock(elapsed)}</span>
            {total > 0 && (
              <span className="loader-found">
                {verified} verified finding{verified === 1 ? "" : "s"} so far
              </span>
            )}
          </div>
        </div>
      </div>
      <ol className="stages">
        {STAGES.map((s, i) => (
          <li key={s} className={i < stage ? "st-done" : i === stage ? "st-now" : ""}>
            <span className="st-dot" />
            {s}
          </li>
        ))}
      </ol>
      {total > 0 && (
        <div className="scene-grid">
          {order.map((s) => (
            <span key={s} className={`scene scene-${state[s]}${stage === 2 ? " scene-scan" : ""}`}
              title={`scene ${s}: ${state[s]}`}>
              {s}
            </span>
          ))}
        </div>
      )}
      {total > 0 && (
        <div className="legend muted small">
          <span><i className="scene-found" /> has a verified finding</span>
          <span><i className="scene-none" /> nothing verified</span>
          <span><i className="scene-pending" /> waiting</span>
        </div>
      )}
      <p className="muted small loader-why">
        Why it takes a while: every claim is checked against the LiDAR before you see it, about {MIN_PER_SCENE} min a
        scene on a laptop. Scenes already verified for this query load at once, and the recorded searches under
        "saved searches" open instantly.
      </p>
    </div>
  );
}

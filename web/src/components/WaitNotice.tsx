import { useEffect, useState } from "react";
import type { TraceEvent } from "../types";

// 5 min a scene (A-249): measured on a laptop with 4 SAM workers (about 2.5 h over the 29 index scenes)
const MIN_PER_SCENE = 5;

function clock(s: number): string {
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m}:${String(s % 60).padStart(2, "0")}`;
}

function minutes(n: number): string {
  const m = n * MIN_PER_SCENE;
  return m >= 90 ? `about ${(m / 60).toFixed(1)} h` : `about ${m} min`;
}

/** While a search runs: what it is doing, how long it should take, and why. */
export function WaitNotice({ events, startedAt }: { events: TraceEvent[]; startedAt: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const elapsed = Math.max(0, Math.round((now - startedAt) / 1000));
  const picked = events
    .filter((e) => e.name === "searching")
    .reduce((n, e) => n + ((e.detail.scene_ids as string[] | undefined)?.length ?? 0), 0);
  const searchingNow = events.some((e) => e.kind === "model" && (e.detail.tool_calls as string[] | undefined)?.includes("search_scenes"));

  let status: string;
  if (!events.some((e) => e.kind === "plan")) status = "Planning: Muse Wonder is turning your sentence into a scenario spec (usually under a minute).";
  else if (picked > 0) status = `Verifying ${picked} scene${picked === 1 ? "" : "s"}: estimated ${minutes(picked)} on a laptop.`;
  else if (searchingNow) status = "Verifying the scenes the agent picked: about 5 min per scene on a laptop (up to 29 scenes, about 2.5 h).";
  else status = "The agent is choosing which scenes to search.";

  return (
    <div className="wait">
      <div className="wait-head">
        <span>{status}</span>
        <span className="wait-clock">elapsed {clock(elapsed)}</span>
      </div>
      <p className="muted small">
        Why it takes a while: for every scene, SAM 3.1 tracks each named object through all six cameras, and the
        verifier checks every track against the LiDAR, frame by frame, before anything is shown. Results appear when
        the agent finishes. Scenes already verified for this query load at once, and the recorded searches under
        "saved searches" open instantly.
      </p>
    </div>
  );
}

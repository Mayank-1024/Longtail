import { useEffect, useState } from "react";

/** Elapsed time as m:ss, or "h min" past an hour. */
export function clock(s: number): string {
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m}:${String(s % 60).padStart(2, "0")}`;
}

/** Seconds since `since`, ticking once a second. */
export function useElapsed(since: number): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  return Math.max(0, Math.round((now - since) / 1000));
}

/** A progress ring: spins while the amount of work is unknown, fills with done/total once it is known. */
export function Ring({ done, total, spinning, center, label }: {
  done: number;
  total: number;
  spinning: boolean;
  center?: string;
  label?: string;
}) {
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
      <text x="42" y="40" className="ring-num">{center ?? (total > 0 ? `${done}/${total}` : "…")}</text>
      <text x="42" y="55" className="ring-label">{label ?? (total > 0 ? "scenes" : "")}</text>
    </svg>
  );
}

export function Stages({ names, now }: { names: readonly string[]; now: number }) {
  return (
    <ol className="stages">
      {names.map((s, i) => (
        <li key={s} className={i < now ? "st-done" : i === now ? "st-now" : ""}>
          <span className="st-dot" />
          {s}
        </li>
      ))}
    </ol>
  );
}

/**
 * The loader for any wait other than a search: what is loading, the step it is on, a count when one is
 * known (otherwise a spinning ring with the elapsed seconds), and why it takes a while. It stays hidden for
 * the first `delayMs` so a quick load doesn't flash it.
 */
export function StepLoader({ title, sub, stages, stage, done, total, unit, why, compact, delayMs = 250 }: {
  title: string;
  sub?: string;
  stages?: readonly string[];
  stage?: number;
  done?: number;
  total?: number;
  unit?: string;
  why?: string;
  compact?: boolean;
  delayMs?: number;
}) {
  const [startedAt] = useState(Date.now());
  const elapsed = useElapsed(startedAt);
  const [shown, setShown] = useState(delayMs <= 0);
  useEffect(() => {
    if (shown) return;
    const t = setTimeout(() => setShown(true), delayMs);
    return () => clearTimeout(t);
  }, [delayMs, shown]);
  if (!shown) return null;
  const counted = total !== undefined && total > 0;
  return (
    <div className={`loader${compact ? " loader-compact" : ""}`} role="status" aria-live="polite">
      <div className="loader-top">
        <Ring done={done ?? 0} total={total ?? 0} spinning={!counted}
          center={counted ? undefined : `${elapsed}s`} label={counted ? unit ?? "" : "elapsed"} />
        <div className="loader-text">
          <div className="loader-head">{title}</div>
          {sub && <div className="loader-sub">{sub}</div>}
          {counted && <div className="loader-meta"><span>elapsed {clock(elapsed)}</span></div>}
        </div>
      </div>
      {stages && <Stages names={stages} now={stage ?? 0} />}
      {why && <p className="muted small loader-why">{why}</p>}
    </div>
  );
}

/** Placeholder rows that shimmer while a list loads. */
export function SkeletonRows({ n = 3, label }: { n?: number; label: string }) {
  return (
    <>
      {Array.from({ length: n }, (_, i) => (
        <li key={i} className="skeleton" aria-hidden="true">
          <span className="sk-line" />
          <span className="sk-line sk-short" />
        </li>
      ))}
      <li className="muted small" role="status">{label}</li>
    </>
  );
}

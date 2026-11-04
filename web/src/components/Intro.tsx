import { useEffect, useRef, useState } from "react";

const WORD = "LongTail";
// Points along the curve where a few "rare events" light up after the tail is drawn (x positions, 0..1000).
const EVENTS = [610, 780, 930];

/** y of the long-tail curve at x: a steep head on the left that thins into a long, low tail. */
function tailY(x: number): number {
  return 160 - 140 * Math.exp(-x / 110);
}
const PATH = Array.from({ length: 101 }, (_, i) => {
  const x = i * 10;
  return `${i === 0 ? "M" : "L"}${x},${tailY(x).toFixed(1)}`;
}).join(" ");

/**
 * The landing screen: the wordmark assembles letter by letter over a LiDAR sweep while the long-tail curve
 * draws beneath it and the rare events at its far end light up green. "Enter LongTail" (or Enter) scatters
 * the letters and zooms through into the app, which has been loading underneath the whole time.
 */
export function Intro({ onEnter }: { onEnter: () => void }) {
  const [leaving, setLeaving] = useState(false);
  const done = useRef(false);
  const btn = useRef<HTMLButtonElement>(null);

  const finish = () => {
    if (done.current) return;
    done.current = true;
    onEnter();
  };
  const enter = () => {
    if (leaving) return;
    setLeaving(true);
    setTimeout(finish, 1000); // the exit animation is 0.9 s; finish even if animationend never fires
  };

  useEffect(() => {
    const t = setTimeout(() => btn.current?.focus({ preventScroll: true }), 1800);
    const k = (e: KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); enter(); } };
    window.addEventListener("keydown", k);
    return () => { clearTimeout(t); window.removeEventListener("keydown", k); };
  });

  return (
    <div className={`intro${leaving ? " intro-leave" : ""}`} role="dialog" aria-label="LongTail"
      onAnimationEnd={(e) => { if (leaving && e.target === e.currentTarget) finish(); }}>
      <div className="intro-sweep" aria-hidden="true">
        <div className="sweep-rings" />
        <div className="sweep-beam" />
      </div>
      <div className="intro-grid" aria-hidden="true" />

      <div className="intro-center">
        <div className="intro-eyebrow">Meta Global AI Developer Hackathon</div>
        <h1 className="intro-word" aria-label={WORD}>
          {WORD.split("").map((c, i) => (
            <span key={i} className={i >= 4 ? "w-tail" : ""}
              style={{ ["--i" as string]: i, ["--dx" as string]: `${(i - 3.5) * 38}px`, ["--rot" as string]: `${(i % 2 ? 1 : -1) * (8 + i * 3)}deg` }}
              aria-hidden="true">
              {c}
            </span>
          ))}
        </h1>

        <svg className="intro-curve" viewBox="0 0 1000 180" aria-hidden="true">
          <defs>
            <linearGradient id="lt-g" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0" stopColor="#38bdf8" />
              <stop offset="0.75" stopColor="#38bdf8" />
              <stop offset="1" stopColor="#22c55e" />
            </linearGradient>
            <linearGradient id="lt-fill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="rgba(56,189,248,0.18)" />
              <stop offset="1" stopColor="rgba(56,189,248,0)" />
            </linearGradient>
          </defs>
          <path className="curve-fill" d={`${PATH} L1000,180 L0,180 Z`} fill="url(#lt-fill)" />
          <path className="curve-line" d={PATH} pathLength={1} stroke="url(#lt-g)" />
          {EVENTS.map((x, i) => (
            <g key={x} className="curve-event" style={{ ["--j" as string]: i }}>
              <circle cx={x} cy={tailY(x)} r={16} className="ev-ring" />
              <circle cx={x} cy={tailY(x)} r={6.5} className="ev-dot" />
            </g>
          ))}
        </svg>

        <p className="intro-tag">
          Search hours of driving logs in plain English.<br />
          Every result is checked against the LiDAR before you see it.
        </p>
        <button ref={btn} className="intro-btn" onClick={enter} disabled={leaving}>
          <span>Enter LongTail</span>
          <span className="intro-arrow" aria-hidden="true">→</span>
        </button>
        <div className="intro-foot">Muse Wonder · SAM 3.1 · Muse Spark Open · PandaSet LiDAR</div>
      </div>
    </div>
  );
}

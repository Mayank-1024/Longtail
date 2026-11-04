import { useEffect, useState } from "react";
import { api } from "./api";
import { Detail } from "./components/Detail";
import { Intro } from "./components/Intro";
import { SkeletonRows, StepLoader } from "./components/Loader";
import { Results } from "./components/Results";
import { TraceView } from "./components/TraceView";
import { Welcome } from "./components/Welcome";
import { WaitNotice } from "./components/WaitNotice";
import type { Search, SearchListItem, TraceEvent } from "./types";

const EXAMPLES = [
  "a pedestrian crossing in front of us while we move",
  "a cyclist within 15 m",
  "a pedestrian partly hidden by a car",
];

// Show the landing screen once per browser tab; a reload in the same tab goes straight to the app.
function introSeen(): boolean {
  try {
    return new URLSearchParams(window.location.search).has("app") || sessionStorage.getItem("lt-entered") === "1";
  } catch {
    return false;
  }
}

export function App() {
  const [intro, setIntro] = useState(() => !introSeen());
  const [entered, setEntered] = useState(false); // true once the intro hands over: plays the app's entrance
  const [mode, setMode] = useState<string>("");
  const [saved, setSaved] = useState<SearchListItem[]>([]);
  const [query, setQuery] = useState("");
  const [live, setLive] = useState<TraceEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<Search | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [savedLoaded, setSavedLoaded] = useState(false);
  const [opening, setOpening] = useState<string | null>(null); // the saved search being opened

  const refresh = () =>
    api.searches().then(setSaved).catch(() => undefined).finally(() => setSavedLoaded(true));
  const show = (sid: string) => {
    setError(null);
    setOpening(sid);
    return api.search(sid)
      .then((s) => { setSearch(s); setOpen(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setOpening(null));
  };
  // after a review: refresh the search in place, so the open finding stays on screen
  const reload = (sid: string) => api.search(sid).then(setSearch).catch((e) => setError(String(e)));

  useEffect(() => {
    api.health().then((h) => setMode(h.mode)).catch(() => setMode("offline"));
    refresh();
  }, []);

  const run = async (q: string) => {
    if (!q.trim()) return;
    setError(null);
    setSearch(null);
    setOpen(null);
    setLive([]);
    setStartedAt(Date.now());
    try {
      const sid = await api.start(q.trim(), (e) => setLive((prev) => [...(prev ?? []), e]));
      await show(sid);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setLive(null);
    }
  };

  const enter = () => {
    try { sessionStorage.setItem("lt-entered", "1"); } catch { /* private window: show it again next time */ }
    setIntro(false);
    setEntered(true);
  };

  return (
    <>
    {intro && <Intro onEnter={enter} />}
    <div className={`app${entered ? " app-enter" : ""}`} aria-hidden={intro || undefined}>
      <header>
        <span className="logo" title="replay the intro" onClick={() => setIntro(true)}>
          <img src="/favicon.svg" alt="" width="22" height="22" />
          LongTail
        </span>
        <span className="tagline">search driving logs in plain English; every result checked on LiDAR</span>
        <span className="spacer" />
        {mode && <span className={`mode mode-${mode}`}>{mode}</span>}
      </header>
      {mode === "offline-demo" && (
        <div className="banner">
          Offline demo: synthetic scenes and a scripted model. The verifier is real, but these are not system results.
        </div>
      )}
      <div className="body">
        <aside>
          <form onSubmit={(e) => { e.preventDefault(); run(query); }}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="describe a scenario…"
              disabled={live !== null} />
            <button className="primary" disabled={live !== null || !query.trim()}>search</button>
          </form>
          <div className="examples">
            {EXAMPLES.map((x) => (
              <button key={x} className="link" onClick={() => { setQuery(x); run(x); }} disabled={live !== null}>
                {x}
              </button>
            ))}
          </div>
          <h3>saved searches</h3>
          <ul className="saved">
            {saved.map((s) => (
              <li key={s.search_id}
                className={`${search?.search_id === s.search_id ? "on" : ""}${opening === s.search_id ? " opening" : ""}`}
                onClick={() => show(s.search_id)}>
                <div>{s.query}</div>
                <div className="muted small">
                  {s.verified} verified · {s.rejected} rejected · {s.stop_reason} · ${s.usd.toFixed(3)}
                </div>
              </li>
            ))}
            {!savedLoaded && <SkeletonRows label="loading saved searches…" />}
            {savedLoaded && saved.length === 0 && <li className="muted small">none yet</li>}
          </ul>
        </aside>
        <main>
          {error && <div className="error">{error}</div>}
          {live !== null && (
            <div>
              <h2>searching…</h2>
              <WaitNotice events={live} startedAt={startedAt} mode={mode} />
              <TraceView events={live} running />
            </div>
          )}
          {live === null && opening && (
            <StepLoader title="Opening the saved search"
              sub={saved.find((x) => x.search_id === opening)?.query}
              why="Reading its findings, trace and reviews from disk: no model call and no cost." />
          )}
          {live === null && !opening && search && !open && <Results s={search} onOpen={setOpen} />}
          {live === null && !opening && search && open && (
            <Detail sid={search.search_id} fid={open} onBack={() => setOpen(null)}
              onChanged={() => reload(search.search_id)} />
          )}
          {live === null && !opening && !search && !error && (
            <Welcome examples={EXAMPLES} disabled={live !== null} onTry={(x) => { setQuery(x); run(x); }} />
          )}
        </main>
      </div>
    </div>
    </>
  );
}

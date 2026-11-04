import { useEffect, useState } from "react";
import { api } from "./api";
import { Detail } from "./components/Detail";
import { Results } from "./components/Results";
import { TraceView } from "./components/TraceView";
import { WaitNotice } from "./components/WaitNotice";
import type { Search, SearchListItem, TraceEvent } from "./types";

const EXAMPLES = [
  "a pedestrian crossing in front of us while we move",
  "a cyclist within 15 m",
  "a pedestrian partly hidden by a car",
];

export function App() {
  const [mode, setMode] = useState<string>("");
  const [saved, setSaved] = useState<SearchListItem[]>([]);
  const [query, setQuery] = useState("");
  const [live, setLive] = useState<TraceEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<Search | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState(0);

  const refresh = () => api.searches().then(setSaved).catch(() => undefined);
  const show = (sid: string) => api.search(sid).then((s) => { setSearch(s); setOpen(null); }).catch((e) => setError(String(e)));

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

  return (
    <div className="app">
      <header>
        <span className="logo">LongTail</span>
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
            <button disabled={live !== null || !query.trim()}>search</button>
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
              <li key={s.search_id} className={search?.search_id === s.search_id ? "on" : ""}
                onClick={() => show(s.search_id)}>
                <div>{s.query}</div>
                <div className="muted small">
                  {s.verified} verified · {s.rejected} rejected · {s.stop_reason} · ${s.usd.toFixed(3)}
                </div>
              </li>
            ))}
            {saved.length === 0 && <li className="muted small">none yet</li>}
          </ul>
        </aside>
        <main>
          {error && <div className="error">{error}</div>}
          {live !== null && (
            <div>
              <h2>searching…</h2>
              <WaitNotice events={live} startedAt={startedAt} />
              <TraceView events={live} running />
            </div>
          )}
          {live === null && search && !open && <Results s={search} onOpen={setOpen} />}
          {live === null && search && open && (
            <Detail sid={search.search_id} fid={open} onBack={() => setOpen(null)}
              onChanged={() => show(search.search_id).then(() => setOpen(open))} />
          )}
          {live === null && !search && !error && (
            <div className="hello">
              <p>Ask for a scenario: the planner turns it into checkable predicates, SAM 3.1 tracks the objects,
                and the LiDAR verifier marks every claim PASS, FAIL or UNVERIFIABLE. Only verified findings are
                results; everything else is shown under "rejected by verifier".</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

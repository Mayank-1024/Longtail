import { useState } from "react";
import { api } from "../api";
import type { Finding, Search } from "../types";
import { StatusBadge } from "./CardView";
import { TraceView } from "./TraceView";

type Tab = "verified" | "rejected" | "trace";

function failing(f: Finding): string {
  const bad = f.results.filter((r) => r.required && r.status !== "PASS");
  return bad.map((r) => `${r.name}: ${r.status.toLowerCase()} (${r.evidence.reason ?? "?"})`).join(" · ");
}

function SpecSummary({ s }: { s: Search }) {
  const spec = s.spec;
  if (spec.clarification) return <div className="clarify">The planner asks: {spec.clarification}</div>;
  return (
    <div className="spec">
      {spec.actors.map((a) => (
        <span key={a.id} className="chip">{a.id}: {a.cls}</span>
      ))}
      {spec.predicates.map((p, i) => (
        <span key={i} className="chip chip-pred">
          {p.name}({[...p.actors, ...Object.entries(p.args).map(([k, v]) => `${k}=${v}`)].join(", ")})
        </span>
      ))}
      {spec.assumptions.map((a) => (
        <span key={a} className="chip chip-assume">{a}</span>
      ))}
    </div>
  );
}

export function Results({ s, onOpen }: { s: Search; onOpen: (fid: string) => void }) {
  const [tab, setTab] = useState<Tab>("verified");
  const list = tab === "verified" ? s.findings : s.rejected;
  const approved = Object.values(s.reviews).filter((r) => r.decision === "approve").length;
  return (
    <div className="results fade-in">
      <h2>{s.query}</h2>
      <SpecSummary s={s} />
      <div className="stats-row">
        <div className="stat stat-pass"><b>{s.findings.length}</b><span>verified</span></div>
        <div className="stat stat-rej"><b>{s.rejected.length}</b><span>rejected by verifier</span></div>
        <div className="stat"><b>{s.scenes_searched.length}</b><span>scenes searched</span></div>
        <div className="stat"><b>{approved}</b><span>approved</span></div>
        <div className="stat"><b>${s.usd.toFixed(2)}</b><span>model spend</span></div>
      </div>
      <div className="muted results-meta">
        stopped: {s.stop_reason}
        <span className="spacer" />
        <a className="btn-link" href={api.exportUrl(s.search_id, "json")}>export JSON</a>
        <a className="btn-link" href={api.exportUrl(s.search_id, "csv")}>export CSV</a>
      </div>
      {s.summary && <p className="summary">{s.summary}</p>}
      <div className="tabs">
        <button className={tab === "verified" ? "on" : ""} onClick={() => setTab("verified")}>
          verified <span className="count count-pass">{s.findings.length}</span>
        </button>
        <button className={tab === "rejected" ? "on" : ""} onClick={() => setTab("rejected")}>
          rejected by verifier <span className="count">{s.rejected.length}</span>
        </button>
        <button className={tab === "trace" ? "on" : ""} onClick={() => setTab("trace")}>
          trace <span className="count">{s.trace.length}</span>
        </button>
      </div>
      {tab === "trace" ? (
        <TraceView events={s.trace} />
      ) : list.length === 0 ? (
        <div className="muted empty">
          {tab === "verified" ? "No finding passed verification." : "Nothing was rejected."}
        </div>
      ) : (
        <ul className="findings">
          {list.map((f, i) => {
            const review = s.reviews[f.finding_id];
            return (
              <li key={f.finding_id} className={`rise ${f.verified ? "f-ok" : "f-no"}`}
                style={{ animationDelay: `${Math.min(i, 12) * 35}ms` }}
                tabIndex={0} onClick={() => onOpen(f.finding_id)}
                onKeyDown={(e) => { if (e.key === "Enter") onOpen(f.finding_id); }}>
                <div className="finding-head">
                  <b>scene {f.scene_id}</b>
                  <span className="muted">{(f.t1 - f.t0).toFixed(1)} s · score {f.score.toFixed(2)}</span>
                  {review && <span className={`review review-${review.decision}`}>{review.decision}d</span>}
                  <span className="spacer" />
                  <span className="open-hint">open →</span>
                </div>
                <div className="finding-results">
                  {f.results.map((r) => (
                    <span key={r.result_id} title={r.evidence.reason}>
                      {r.name} <StatusBadge status={r.status} />
                    </span>
                  ))}
                </div>
                {!f.verified && <div className="why">why: {failing(f)}</div>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

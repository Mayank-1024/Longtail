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
    <div className="results">
      <h2>{s.query}</h2>
      <SpecSummary s={s} />
      <div className="muted">
        {s.scenes_searched.length} scene(s) searched · stopped: {s.stop_reason} · ${s.usd.toFixed(4)} · {approved} approved
        <span className="spacer" />
        <a href={api.exportUrl(s.search_id, "json")}>export JSON</a>
        <a href={api.exportUrl(s.search_id, "csv")}>export CSV</a>
      </div>
      {s.summary && <p className="summary">{s.summary}</p>}
      <div className="tabs">
        <button className={tab === "verified" ? "on" : ""} onClick={() => setTab("verified")}>
          verified ({s.findings.length})
        </button>
        <button className={tab === "rejected" ? "on" : ""} onClick={() => setTab("rejected")}>
          rejected by verifier ({s.rejected.length})
        </button>
        <button className={tab === "trace" ? "on" : ""} onClick={() => setTab("trace")}>
          trace ({s.trace.length})
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
          {list.map((f) => {
            const review = s.reviews[f.finding_id];
            return (
              <li key={f.finding_id} onClick={() => onOpen(f.finding_id)}>
                <div className="finding-head">
                  <b>scene {f.scene_id}</b>
                  <span className="muted">{(f.t1 - f.t0).toFixed(1)} s · score {f.score.toFixed(2)}</span>
                  {review && <span className={`review review-${review.decision}`}>{review.decision}d</span>}
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

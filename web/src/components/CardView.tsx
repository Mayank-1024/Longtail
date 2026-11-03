import type { Card, PredicateResult } from "../types";

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge badge-${status.toLowerCase()}`}>{status}</span>;
}

/** The evidence card: every sentence shows the verifier results it cites. */
export function CardView({ card, results }: { card: Card; results: PredicateResult[] }) {
  const byId = new Map(results.map((r) => [r.result_id, r]));
  return (
    <div className="card">
      <div className="card-title">
        {card.title}
        <span className="card-source" title="who wrote the card">
          {card.source === "model" ? "Spark Open" : "template"}
        </span>
      </div>
      <ul className="card-sentences">
        {card.sentences.map((s, i) => (
          <li key={i}>
            {s.text}{" "}
            {s.cites.map((c) => {
              const r = byId.get(c);
              return (
                <span
                  key={c}
                  className={`cite cite-${(r?.status ?? "unknown").toLowerCase()}`}
                  title={r ? `${r.name}: ${r.status} (${r.evidence.reason ?? ""})` : c}
                >
                  {r ? r.name : "?"}
                </span>
              );
            })}
          </li>
        ))}
      </ul>
      {card.note && <div className="card-note">{card.note}</div>}
    </div>
  );
}

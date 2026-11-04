import type { TraceEvent } from "../types";

function summary(e: TraceEvent): string {
  const d = e.detail as Record<string, any>;
  if (e.kind === "plan") {
    const spec = d.spec ?? {};
    if (spec.clarification) return `asks: ${spec.clarification}`;
    const preds = (spec.predicates ?? []).map((p: any) => p.name).join(", ");
    return `${(spec.actors ?? []).length} actor(s); ${preds} (${d.attempts} attempt${d.attempts === 1 ? "" : "s"})`;
  }
  if (e.kind === "model") return `calls ${(d.tool_calls ?? []).join(", ") || "nothing"}`;
  if (e.kind === "stop") {
    if (e.name === "finish_dropped") return `dropped ${(d.finding_ids ?? []).length} unverified id(s) named in finish`;
    return d.error ? String(d.error) : e.name.replace("_", " ");
  }
  if (e.name === "searching") return `verifying ${((d.scene_ids ?? []) as string[]).length} scene(s) on LiDAR…`;
  if (e.name === "list_scenes") return `${(d.result?.scenes ?? []).length} scenes available`;
  if (e.name === "search_scenes") {
    const scenes = d.result?.scenes ?? [];
    const v = scenes.reduce((n: number, s: any) => n + (s.verified ?? 0), 0);
    const u = scenes.reduce((n: number, s: any) => n + (s.unverified ?? 0), 0);
    const errs = scenes.filter((s: any) => s.error).length;
    return `searched ${scenes.length} scene(s): ${v} verified, ${u} rejected${errs ? `, ${errs} failed` : ""}`;
  }
  if (e.name === "finish") return `finish with ${(d.finding_ids ?? []).length} finding(s)`;
  return JSON.stringify(d).slice(0, 120);
}

export function TraceView({ events, running }: { events: TraceEvent[]; running?: boolean }) {
  const total = events.reduce((n, e) => n + (e.usd ?? 0), 0);
  return (
    <div className="trace">
      <ol>
        {events.map((e, i) => (
          <li key={i} className={`trace-${e.kind}`}>
            <span className="trace-step">{e.step}</span>
            <span className="trace-kind">{e.kind}</span>
            <span className="trace-name">{e.name.replace("_", " ")}</span>
            <span className="trace-summary">{summary(e)}</span>
            <span className="trace-usd">{e.usd ? `$${e.usd.toFixed(4)}` : ""}</span>
          </li>
        ))}
        {running && <li className="trace-running">working…</li>}
      </ol>
      <div className="trace-total">model spend this search: ${total.toFixed(4)}</div>
    </div>
  );
}

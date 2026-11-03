import type { Bev, Card, FindingDetail, Review, Search, SearchListItem, TraceEvent } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<{ ok: boolean; mode: string }>(r)),
  searches: () => fetch("/api/searches").then((r) => json<SearchListItem[]>(r)),
  search: (sid: string) => fetch(`/api/searches/${sid}`).then((r) => json<Search>(r)),
  finding: (sid: string, fid: string) =>
    fetch(`/api/searches/${sid}/findings/${fid}`).then((r) => json<FindingDetail>(r)),
  writeCard: (sid: string, fid: string) =>
    fetch(`/api/searches/${sid}/findings/${fid}/card`, { method: "POST" }).then((r) => json<Card>(r)),
  review: (sid: string, finding_id: string, decision: "approve" | "reject", note = "") =>
    fetch(`/api/searches/${sid}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ finding_id, decision, note }),
    }).then((r) => json<Review>(r)),
  bev: (url: string) => fetch(url).then((r) => json<Bev>(r)),
  exportUrl: (sid: string, format: "json" | "csv") => `/api/searches/${sid}/export?format=${format}`,

  /** Start a search and stream its trace. Resolves with the search id, or rejects with the server's error. */
  start(query: string, onEvent: (e: TraceEvent) => void): Promise<string> {
    return fetch("/api/searches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    })
      .then((r) => json<{ job_id: string }>(r))
      .then(
        ({ job_id }) =>
          new Promise<string>((resolve, reject) => {
            const es = new EventSource(`/api/jobs/${job_id}/events`);
            es.addEventListener("trace", (m) => onEvent(JSON.parse((m as MessageEvent).data)));
            es.addEventListener("done", (m) => {
              es.close();
              const d = JSON.parse((m as MessageEvent).data);
              if (d.status === "done" && d.search_id) resolve(d.search_id);
              else reject(new Error(d.error ?? "the search failed"));
            });
            es.onerror = () => {
              es.close();
              reject(new Error("lost the connection to the trace stream"));
            };
          }),
      );
  },
};

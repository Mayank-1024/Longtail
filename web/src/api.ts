import type { Bev, Card, FindingDetail, Review, Search, SearchListItem, TraceEvent } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** The optional API token (LONGTAIL_API_TOKEN): open the web view as /?token=..., kept for this tab. */
function apiToken(): string | null {
  try {
    const fromUrl = new URLSearchParams(window.location.search).get("token");
    if (fromUrl) sessionStorage.setItem("longtail-token", fromUrl);
    return fromUrl ?? sessionStorage.getItem("longtail-token");
  } catch {
    return null;
  }
}

function post(url: string, body?: unknown): Promise<Response> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = apiToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return fetch(url, { method: "POST", headers, body: body === undefined ? undefined : JSON.stringify(body) });
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<{ ok: boolean; mode: string }>(r)),
  searches: () => fetch("/api/searches").then((r) => json<SearchListItem[]>(r)),
  search: (sid: string) => fetch(`/api/searches/${sid}`).then((r) => json<Search>(r)),
  finding: (sid: string, fid: string) =>
    fetch(`/api/searches/${sid}/findings/${fid}`).then((r) => json<FindingDetail>(r)),
  writeCard: (sid: string, fid: string) =>
    post(`/api/searches/${sid}/findings/${fid}/card`).then((r) => json<Card>(r)),
  review: (sid: string, finding_id: string, decision: "approve" | "reject", note = "") =>
    post(`/api/searches/${sid}/reviews`, { finding_id, decision, note }).then((r) => json<Review>(r)),
  bev: (url: string) => fetch(url).then((r) => json<Bev>(r)),
  exportUrl: (sid: string, format: "json" | "csv") => `/api/searches/${sid}/export?format=${format}`,

  /** Start a search and stream its trace. Resolves with the search id, or rejects with the server's error. */
  start(query: string, onEvent: (e: TraceEvent) => void): Promise<string> {
    return post("/api/searches", { query })
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

// Shapes returned by api/server.py. Finding and spec follow longtail/contracts.py; viz follows docs/viz-schema.md.

export type Status = "PASS" | "FAIL" | "UNVERIFIABLE";
export type Mode = "live" | "replay" | "offline-demo";

export interface PredicateResult {
  result_id: string;
  name: string;
  actors: Record<string, string>;
  status: Status;
  value: number | null;
  threshold: string;
  required: boolean;
  evidence: {
    reason?: string;
    margin?: number;
    longest_pass_s?: number;
    frame_status?: Record<string, number>;
    example?: Record<string, unknown>;
    [k: string]: unknown;
  };
}

export interface Finding {
  finding_id: string;
  scene_id: string;
  t0: number;
  t1: number;
  bindings: Record<string, string>;
  results: PredicateResult[];
  verified: boolean;
  score: number;
  cost_usd: number;
}

export interface Spec {
  query: string;
  actors: { id: string; cls: string; sam_prompt: string }[];
  predicates: { name: string; actors: string[]; args: Record<string, number | string>; required: boolean }[];
  min_duration_s: number;
  assumptions: string[];
  clarification: string | null;
}

export interface TraceEvent {
  step: number;
  kind: "plan" | "model" | "tool" | "stop";
  name: string;
  detail: Record<string, unknown>;
  usd: number;
  t: number;
}

export interface CardSentence {
  text: string;
  cites: string[];
}

export interface Card {
  title: string;
  sentences: CardSentence[];
  source: "model" | "template";
  note?: string | null;
}

export interface Review {
  finding_id: string;
  decision: "approve" | "reject";
  note: string;
  at: number;
}

export interface Search {
  search_id: string;
  query: string;
  spec: Spec;
  findings: Finding[];
  rejected: Finding[];
  summary: string;
  stop_reason: string;
  usd: number;
  scenes_searched: string[];
  trace: TraceEvent[];
  reviews: Record<string, Review>;
  mode: Mode;
}

export interface SearchListItem {
  search_id: string;
  query: string;
  stop_reason: string;
  verified: number;
  rejected: number;
  usd: number;
  scenes_searched: number;
  modified: number;
}

export interface Overlay {
  camera: string;
  frame_idx: number;
  camera_frame_idx: number;
  ts: number;
  path: string;
  url: string;
}

export interface TimelineEntry {
  frame_idx: number;
  ts: number;
  status: Status | "NOT_JUDGED";
  reason: string | null;
}

export interface Timeline {
  finding_id: string;
  verified: boolean;
  t0: number;
  t1: number;
  frames: { frame_idx: number; ts: number; in_interval: boolean }[];
  predicates: {
    result_id: string;
    name: string;
    actors: Record<string, string>;
    required: boolean;
    status: Status;
    reason: string;
    entries: TimelineEntry[];
  }[];
}

export interface Viz {
  finding_id: string;
  verified: boolean;
  overlays: Overlay[];
  missing_overlays: { camera: string; frame_idx: number; reason?: string }[];
  bev_url: string;
  bev_bytes: number;
  timeline: Timeline;
}

export interface FindingDetail {
  finding: Finding;
  spec: Spec;
  card: Card;
  review: Review | null;
  viz: Viz | null;
  mode: Mode;
}

export type Vec2 = [number, number];

export interface BevTrack {
  track_id: string;
  cls: string;
  actors: string[];
  position_xyz: [number, number, number] | null;
  box: { center_xy: Vec2; yaw_rad: number; length_m: number; width_m: number } | null;
  speed_mps: number | null;
  velocity_xy_mps: Vec2 | null;
  speed_reliable: boolean;
}

export interface BevFrame {
  frame_idx: number;
  ts: number;
  points_xyz: [number, number, number][];
  ego_box: Vec2[];
  corridor: Vec2[];
  corridor_centerline: Vec2[];
  corridor_extended: boolean;
  tracks: BevTrack[];
}

export interface Bev {
  frame: "ego";
  frames: BevFrame[];
}

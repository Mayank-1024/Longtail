import type { BevFrame, Vec2 } from "../types";

// Ego frame: x forward, y left. On screen: forward is up, left is left. SVG units are metres.
const sx = (p: Vec2) => -p[1];
const sy = (p: Vec2) => -p[0];
const pts = (ps: Vec2[]) => ps.map((p) => `${sx(p)},${sy(p)}`).join(" ");
const ACTOR = ["#22d3ee", "#e879f9", "#4ade80", "#fb923c", "#a78bfa"];

function boxCorners(c: Vec2, yaw: number, l: number, w: number): Vec2[] {
  const [cx, cy] = c;
  const ca = Math.cos(yaw);
  const sa = Math.sin(yaw);
  return [
    [l / 2, w / 2],
    [l / 2, -w / 2],
    [-l / 2, -w / 2],
    [-l / 2, w / 2],
  ].map(([a, b]) => [cx + a * ca - b * sa, cy + a * sa + b * ca] as Vec2);
}

function heightColour(z: number): string {
  // z is height above the LiDAR (about 1.8 m above the road): road dark, tall objects bright.
  const t = Math.max(0, Math.min(1, (z + 1.8) / 3.5));
  return `hsl(${210 - 170 * t}, 70%, ${25 + 45 * t}%)`;
}

export function BevView({ frame, actorOrder }: { frame: BevFrame | undefined; actorOrder: string[] }) {
  if (!frame) return <div className="bev-empty">no bird's-eye data for this frame</div>;
  return (
    <svg className="bev" viewBox="-25 -45 50 58" preserveAspectRatio="xMidYMid meet">
      {[10, 20, 30, 40].map((r) => (
        <circle key={r} cx={0} cy={0} r={r} className="bev-ring" />
      ))}
      {frame.points_xyz.map((p, i) => (
        <circle key={i} cx={-p[1]} cy={-p[0]} r={0.12} fill={heightColour(p[2])} />
      ))}
      {frame.corridor.length > 2 && <polygon points={pts(frame.corridor)} className="bev-corridor" />}
      {frame.corridor_centerline.length > 1 && (
        <polyline points={pts(frame.corridor_centerline)} className="bev-centerline" />
      )}
      <polygon points={pts(frame.ego_box)} className="bev-ego" />
      {frame.tracks.map((t) => {
        const colour = ACTOR[Math.max(0, actorOrder.indexOf(t.actors[0] ?? "")) % ACTOR.length];
        const corners = t.box ? boxCorners(t.box.center_xy, t.box.yaw_rad, t.box.length_m, t.box.width_m) : null;
        const at: Vec2 | null = t.box ? t.box.center_xy : t.position_xyz ? [t.position_xyz[0], t.position_xyz[1]] : null;
        const v = t.velocity_xy_mps;
        return (
          <g key={t.track_id}>
            {corners && <polygon points={pts(corners)} fill="none" stroke={colour} strokeWidth={0.25} />}
            {at && !corners && <circle cx={sx(at)} cy={sy(at)} r={0.5} fill={colour} />}
            {at && v && (
              <line x1={sx(at)} y1={sy(at)} x2={sx([at[0] + v[0], at[1] + v[1]])} y2={sy([at[0] + v[0], at[1] + v[1]])}
                stroke={colour} strokeWidth={0.2} />
            )}
            {at && (
              <text x={sx(at) + 0.8} y={sy(at) - 0.8} className="bev-label" fill={colour}>
                {t.actors.join(",")} {t.speed_mps != null ? `${t.speed_mps.toFixed(1)} m/s` : ""}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

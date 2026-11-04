const STEPS = [
  { n: "01", name: "Plan", model: "Muse Wonder", what: "turns your sentence into actors, relations and durations" },
  { n: "02", name: "Track", model: "SAM 3.1", what: "segments and tracks each named object through six cameras" },
  { n: "03", name: "Verify", model: "LiDAR check", what: "marks every claim PASS, FAIL or UNVERIFIABLE; plain code" },
  { n: "04", name: "Explain", model: "Muse Spark Open", what: "writes an evidence card; each sentence cites a measurement" },
];

/** The empty state: what LongTail does, in four steps, and queries to try with one click. */
export function Welcome({ examples, onTry, disabled }: {
  examples: string[];
  onTry: (q: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="welcome fade-in">
      <div className="eyebrow">search driving logs in plain English</div>
      <h1>Find the rare moments, <span className="hl">checked against LiDAR</span>.</h1>
      <p className="lede">
        Describe a scenario. Only findings the LiDAR check supports become results; everything else is listed
        under "rejected by verifier", with the reason.
      </p>
      <ol className="pipeline">
        {STEPS.map((s) => (
          <li key={s.n}>
            <span className="pl-n">{s.n}</span>
            <span className="pl-name">{s.name}</span>
            <span className="pl-model">{s.model}</span>
            <span className="pl-what">{s.what}</span>
          </li>
        ))}
      </ol>
      <div className="try">
        <div className="try-label">try one</div>
        {examples.map((x) => (
          <button key={x} className="try-q" disabled={disabled} onClick={() => onTry(x)}>
            <span>{x}</span>
            <span className="try-go" aria-hidden="true">→</span>
          </button>
        ))}
      </div>
      <p className="muted small">
        Measured on 15 held-out queries over 102 PandaSet scenes: the LiDAR check raised precision from 0.09 to
        0.20. Live searches take about 5 min a scene; saved searches open at once.
      </p>
    </div>
  );
}

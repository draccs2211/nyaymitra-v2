import type { QueryResponse } from "../lib/api";
import { CARD_HEADINGS, inlineBold, parseAnswer } from "../lib/answer";
import { useI18n } from "../lib/i18n";

function Inline({ text }: { text: string }) {
  return <>{inlineBold(text).map(([s, bold], i) => (bold ? <strong key={i}>{s}</strong> : <span key={i}>{s}</span>))}</>;
}

export function AnswerView({ result }: { result: QueryResponse }) {
  const { t } = useI18n();
  const sections = parseAnswer(result.response).filter((s) => !(s.heading && CARD_HEADINGS.has(s.heading)));
  const cards: Array<[string, string[], string]> = [
    [t.missing, result.missing_facts, "card-amber"],
    [t.uncertain, result.uncertainties, "card-slate"],
    [t.questions, result.questions_for_lawyer, "card-slate"],
  ];

  return (
    <article className="card answer">
      <div className="chips">
        <span className="chip">{t.area}: {result.legal_domain}</span>
      </div>

      {result.is_emergency && (
        <div className="alert alert-danger" role="alert">
          <strong>{t.emergency}{result.urgency_level ? ` (${result.urgency_level})` : ""}</strong>
        </div>
      )}

      {result.unverified_citations.length > 0 && (
        <div className="alert alert-danger" role="alert">
          {t.unverified} <span className="mono">{result.unverified_citations.join("; ")}</span>
        </div>
      )}

      {sections.map((s, i) => (
        <section key={i}>
          {s.heading && <h2 className="h-serif">{s.heading}</h2>}
          {s.kind === "bullets" && <ul>{s.items.map((it, j) => <li key={j}><Inline text={it} /></li>)}</ul>}
          {s.kind === "numbered" && <ol>{s.items.map((it, j) => <li key={j}><Inline text={it} /></li>)}</ol>}
          {s.kind === "text" && s.items.map((it, j) => <p key={j}><Inline text={it} /></p>)}
        </section>
      ))}

      {cards.some(([, items]) => items.length > 0) && (
        <div className="card-grid">
          {cards.filter(([, items]) => items.length > 0).map(([title, items, cls]) => (
            <div key={title} className={`subcard ${cls}`}>
              <h3>{title}{cls === "card-amber" ? ` (${items.length})` : ""}</h3>
              <ul>{items.map((it, j) => <li key={j}>{it}</li>)}</ul>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

export function SourcesPanel({ result }: { result: QueryResponse }) {
  const { t } = useI18n();
  const degraded = Object.entries(result.source_status).filter(([, v]) => v !== "ok");
  return (
    <section className="card">
      <h2 className="h-serif h-small">{t.sources}</h2>
      {result.sources.length ? (
        <ul className="source-list">{result.sources.map((s, i) => <li key={i}>{s}</li>)}</ul>
      ) : (
        <p className="muted">{t.noSources}</p>
      )}
      {degraded.map(([name, v]) => (
        <p key={name} className="muted small">{name.replace(/_/g, " ")}: {t.sourceStatus[v] ?? v}</p>
      ))}
    </section>
  );
}

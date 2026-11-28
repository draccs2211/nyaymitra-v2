import { useEffect, useRef, useState, type FormEvent } from "react";
import { AnswerView, SourcesPanel } from "../components/AnswerView";
import { DraftPanel } from "../components/DraftPanel";
import { Progress } from "../components/Progress";
import { api, ApiError, type CaseRow, type QueryResponse } from "../lib/api";
import { useI18n } from "../lib/i18n";
import { JURISDICTIONS } from "../lib/jurisdictions";

const POLL_MS = 1000;

export function Workspace() {
  const { t, lang } = useI18n();
  const [query, setQuery] = useState("");
  const [jurisdiction, setJurisdiction] = useState("IN");
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [caseId, setCaseId] = useState("");
  const [creating, setCreating] = useState(false);
  const [caseTitle, setCaseTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState("");
  const sessionRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    api.cases().then(setCases).catch(() => setCases([]));
  }, []);

  const saveCase = async () => {
    if (!caseTitle.trim()) return;
    try {
      const c = await api.createCase(caseTitle.trim());
      setCases(await api.cases());
      setCaseId(c.id);
      setCreating(false);
      setCaseTitle("");
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || busy) return;
    const runId = crypto.randomUUID();       // lets us poll progress while the query runs
    setBusy(true);
    setError("");
    setResult(null);
    setStage("screening");
    const timer = setInterval(() => {
      api.runProgress(runId).then((p) => setStage(p.stage)).catch(() => { /* not started yet */ });
    }, POLL_MS);
    try {
      const res = await api.query({
        query: query.trim(), jurisdiction, run_id: runId,
        case_id: caseId || undefined, session_id: sessionRef.current,
      });
      sessionRef.current = res.session_id;  // follow-up questions keep the conversation
      setResult(res);
      setStage("completed");
    } catch (err) {
      setStage(null);
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      clearInterval(timer);
      setBusy(false);
    }
  };

  return (
    <div className="workspace">
      <section className="col-main" aria-label={t.research}>
        <form className="card stack" onSubmit={submit}>
          <label htmlFor="q" className="label-strong">{t.askLabel}</label>
          <textarea id="q" rows={3} value={query} placeholder={t.askPlaceholder} maxLength={4000}
            onChange={(e) => setQuery(e.target.value)} />
          <div className="form-row">
            <div className="field">
              <label htmlFor="jur">{t.state}</label>
              <select id="jur" value={jurisdiction} onChange={(e) => setJurisdiction(e.target.value)}>
                {JURISDICTIONS.map((j) => (
                  <option key={j.code} value={j.code}>{lang === "hi" ? j.hi : j.en} ({j.code})</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="case">{t.case}</label>
              {creating ? (
                <div className="inline">
                  <input id="case" placeholder={t.caseTitle} value={caseTitle} maxLength={300}
                    onChange={(e) => setCaseTitle(e.target.value)} />
                  <button type="button" className="btn-secondary" onClick={() => void saveCase()}>{t.save}</button>
                  <button type="button" className="btn-link" onClick={() => setCreating(false)}>{t.cancel}</button>
                </div>
              ) : (
                <div className="inline">
                  <select id="case" value={caseId} onChange={(e) => setCaseId(e.target.value)}>
                    <option value="">{t.noCase}</option>
                    {cases.map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}
                  </select>
                  <button type="button" className="btn-link" onClick={() => setCreating(true)}>+ {t.newCase}</button>
                </div>
              )}
            </div>
            <button type="submit" className="btn-primary push-right" disabled={busy || !query.trim()}>
              {busy ? t.asking : t.ask}
            </button>
          </div>
        </form>

        {stage && <Progress stage={stage} />}
        {error && <p className="alert alert-danger" role="alert">{error}</p>}
        {result ? <AnswerView result={result} /> : !busy && <p className="card muted">{t.empty}</p>}
      </section>

      {result && (
        <aside className="col-side" aria-label={t.sources}>
          <SourcesPanel result={result} />
          {result.has_document && (
            <DraftPanel
              key={result.run_id ?? ""}
              english={result.document_english}
              hindi={result.document_hindi}
              footer={result.approval_status !== "not_required" && (
                <p className={`status status-${result.approval_status}`}>{t.approval[result.approval_status]}</p>
              )}
            />
          )}
        </aside>
      )}
    </div>
  );
}

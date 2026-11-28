import { useCallback, useEffect, useState } from "react";
import { DraftPanel } from "../components/DraftPanel";
import { api, ApiError, type ApprovalRow, type ApprovalStatus, type QueryResponse } from "../lib/api";
import { canDecide, useAuth } from "../lib/auth";
import { humanize } from "../lib/draft";
import { useI18n } from "../lib/i18n";

const FILTERS: Array<ApprovalStatus | "all"> = ["pending", "approved", "rejected", "expired"];

export function Review() {
  const { t, lang } = useI18n();
  const { user } = useAuth();
  const [filter, setFilter] = useState<ApprovalStatus | "all">("pending");
  const [rows, setRows] = useState<ApprovalRow[] | null>(null);
  const [selected, setSelected] = useState<ApprovalRow | null>(null);
  const [detail, setDetail] = useState<QueryResponse | null>(null);
  const [text, setText] = useState("");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setRows(null);
    api.approvals(filter).then(setRows).catch((e) => { setRows([]); setError((e as Error).message); });
  }, [filter]);

  useEffect(load, [load]);

  const open = async (row: ApprovalRow) => {
    setSelected(row);
    setDetail(null);
    setError("");
    setInfo("");
    setNotes("");
    try {
      const d = await api.approval(row.run_id);
      setDetail(d);
      setText(d.response);
    } catch (e) {
      setError((e as Error).message);
      if (e instanceof ApiError && e.status === 410) load();   // expired: list status changed
    }
  };

  const decide = async (decision: "approved" | "rejected") => {
    if (!selected || !detail) return;
    setBusy(true);
    setError("");
    try {
      await api.decide(selected.run_id, {
        decision,
        notes: notes.trim(),
        edited_response: decision === "approved" && text !== detail.response ? text : undefined,
      });
      setInfo(t.decided);
      setSelected(null);
      setDetail(null);
      load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const fmt = (iso: string | null) =>
    iso ? new Date(iso + (iso.endsWith("Z") ? "" : "Z")).toLocaleString(lang === "hi" ? "hi-IN" : "en-IN",
      { dateStyle: "medium", timeStyle: "short" }) : "";
  const mayDecide = !!(user && selected && selected.status === "pending" && canDecide(user, selected.requester_id));

  return (
    <div className="stack">
      <div className="row-between wrap">
        <div>
          <h1 className="h-serif h-page">{t.reviewTitle}</h1>
          <p className="muted">{t.reviewHint}</p>
        </div>
        <div className="pills" role="group" aria-label={t.filter}>
          {FILTERS.map((f) => (
            <button key={f} type="button" aria-pressed={filter === f} onClick={() => { setFilter(f); setSelected(null); setDetail(null); }}>
              {t.statuses[f]}
            </button>
          ))}
        </div>
      </div>

      {info && <p className="alert alert-ok" role="status">{info}</p>}

      <div className="review">
        <section className="card list-card" aria-label={t.statuses[filter]}>
          {rows === null ? <p className="muted pad">{t.loading}</p>
            : rows.length === 0 ? <p className="muted pad">{t.nothingHere}</p>
              : (
                <ul className="queue">
                  {rows.map((r) => (
                    <li key={r.run_id}>
                      <button type="button" aria-current={selected?.run_id === r.run_id ? "true" : undefined}
                        onClick={() => void open(r)}>
                        <span className="queue-title">{r.document_type ? humanize(r.document_type) : t.answerDoc}</span>
                        <span className="muted small">
                          {r.requester_id === user?.id ? t.you : t.colleague} · {fmt(r.created_at)}
                        </span>
                        {filter !== "pending" && <span className="small">{t.statuses[r.status]}</span>}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
        </section>

        <section className="card stack detail" aria-label={t.review}>
          {error && <p className="alert alert-danger" role="alert">{error}</p>}
          {!selected && !error && <p className="muted">{t.select}</p>}
          {selected && !detail && !error && <p className="muted">{t.loading}</p>}
          {selected && detail && (
            <>
              <h2 className="h-serif">
                {selected.document_type ? humanize(selected.document_type) : t.answerDoc}
              </h2>
              {detail.unverified_citations.length > 0 && (
                <div className="alert alert-danger" role="alert">
                  {t.unverified} <span className="mono">{detail.unverified_citations.join("; ")}</span>
                </div>
              )}
              <label htmlFor="resp" className="label-strong">{t.responseLabel}</label>
              <textarea id="resp" rows={12} value={text} readOnly={!mayDecide}
                onChange={(e) => setText(e.target.value)} />
              {detail.has_document && (
                <DraftPanel key={selected.run_id} english={detail.document_english}
                  hindi={detail.document_hindi} editable={false} />
              )}
              {mayDecide ? (
                <>
                  <div className="field">
                    <label htmlFor="notes">{t.notes}</label>
                    <input id="notes" value={notes} placeholder={t.notesHint} onChange={(e) => setNotes(e.target.value)} />
                  </div>
                  <div className="actions">
                    <button type="button" className="btn-danger" disabled={busy || !notes.trim()}
                      onClick={() => void decide("rejected")}>{t.reject}</button>
                    <button type="button" className="btn-primary" disabled={busy}
                      onClick={() => void decide("approved")}>{t.approve}</button>
                  </div>
                </>
              ) : selected.status === "pending" && <p className="muted">{t.waitingSenior}</p>}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

import { useMemo, useState, type ReactNode } from "react";
import { blanks, fill, segments } from "../lib/draft";
import { useI18n } from "../lib/i18n";

interface Props {
  english: string | null;
  hindi: string | null;
  /** Fill-in fields for blanks; off in the review queue, where the draft is read-only. */
  editable?: boolean;
  footer?: ReactNode;
}

export function DraftPanel({ english, hindi, editable = true, footer }: Props) {
  const { t } = useI18n();
  const [tab, setTab] = useState<"en" | "hi">(english ? "en" : "hi");
  const [values, setValues] = useState<Record<string, string>>({});
  const [copied, setCopied] = useState(false);
  const text = (tab === "en" ? english : hindi) ?? "";
  const labels = useMemo(() => blanks(text), [text]);
  const left = labels.filter((l) => !values[l]?.trim()).length;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(fill(text, values));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard blocked: user can still select the text */
    }
  };

  return (
    <section className="card draft">
      <div className="row-between">
        <h2 className="h-serif h-small">{t.draft}</h2>
        {english && hindi && (
          <div className="tabs" role="tablist" aria-label={t.draftLang}>
            <button type="button" role="tab" aria-selected={tab === "en"} onClick={() => setTab("en")}>English</button>
            <button type="button" role="tab" lang="hi" aria-selected={tab === "hi"} onClick={() => setTab("hi")}>हिंदी</button>
          </div>
        )}
      </div>

      <div className="draft-text" lang={tab === "hi" ? "hi" : "en"}>
        {segments(text, values).map((s, i) =>
          s.blank === null ? <span key={i}>{s.text}</span>
            : s.filled ? <mark key={i} className="blank-filled">{s.text}</mark>
              : <mark key={i} className="blank">{s.text}</mark>,
        )}
      </div>

      {editable && labels.length > 0 && (
        <fieldset className="blanks">
          <legend>{left ? t.blanksLeft(left) : t.allFilled}</legend>
          {labels.map((label, i) => (
            <div key={label} className="field">
              <label htmlFor={`blank-${tab}-${i}`}>{label}</label>
              <input id={`blank-${tab}-${i}`} value={values[label] ?? ""}
                onChange={(e) => setValues({ ...values, [label]: e.target.value })} />
            </div>
          ))}
        </fieldset>
      )}

      {editable && (
        <button type="button" className="btn-secondary" onClick={() => void copy()}>
          {copied ? t.copied : t.copyDraft}
        </button>
      )}
      {footer}
    </section>
  );
}

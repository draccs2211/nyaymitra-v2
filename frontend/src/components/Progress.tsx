import { useI18n } from "../lib/i18n";

// Stage names from STAGES in backend/pipeline.py, in pipeline order.
export const STAGE_ORDER = ["screening", "classifying", "researching", "analysing", "drafting", "verifying"];

/** Index of the stage now running; legal-aid lookup runs between drafting and verification. */
export function stageIndex(stage: string | null): number {
  if (!stage) return 0;
  if (stage === "completed" || stage === "awaiting_approval") return STAGE_ORDER.length;
  if (stage === "finding_legal_aid") return STAGE_ORDER.indexOf("verifying");
  const i = STAGE_ORDER.indexOf(stage);
  return i < 0 ? 0 : i;
}

export function Progress({ stage }: { stage: string | null }) {
  const { t } = useI18n();
  const current = stageIndex(stage);
  return (
    <ol className="progress card" aria-label={t.progress}>
      {STAGE_ORDER.map((s, i) => {
        const state = i < current ? "done" : i === current ? "active" : "todo";
        return (
          <li key={s} className={`step step-${state}`} aria-current={state === "active" ? "step" : undefined}>
            {state === "done" ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
                strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
            ) : (
              <span className="dot" aria-hidden="true" />
            )}
            {stage === "finding_legal_aid" && s === "verifying" ? t.stages.finding_legal_aid : t.stages[s]}
          </li>
        );
      })}
    </ol>
  );
}

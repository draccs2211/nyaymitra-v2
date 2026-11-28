import { useI18n } from "../lib/i18n";

export function LangToggle({ light = false }: { light?: boolean }) {
  const { lang, setLang, t } = useI18n();
  return (
    <div className={light ? "segmented segmented-light" : "segmented"} role="group" aria-label={t.uiLang}>
      <button type="button" aria-pressed={lang === "en"} onClick={() => setLang("en")}>EN</button>
      <button type="button" lang="hi" aria-pressed={lang === "hi"} onClick={() => setLang("hi")}>हिं</button>
    </div>
  );
}

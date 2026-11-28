// Blanks in drafted documents are bracketed placeholders such as [Client Name]
// (backend/agents/drafting.py keeps them instead of inventing facts).

export type Segment = { text: string; blank: string | null };

const PLACEHOLDER = /\[([^\[\]\n]{1,80})\]/g;

/** Unique placeholder labels in order of first appearance. */
export function blanks(text: string): string[] {
  const seen: string[] = [];
  for (const m of text.matchAll(PLACEHOLDER)) {
    if (!seen.includes(m[1])) seen.push(m[1]);
  }
  return seen;
}

/** Text split into plain runs and blanks; a blank with a value becomes plain text marked filled. */
export function segments(text: string, values: Record<string, string> = {}): Array<Segment & { filled: boolean }> {
  const out: Array<Segment & { filled: boolean }> = [];
  let last = 0;
  for (const m of text.matchAll(PLACEHOLDER)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start), blank: null, filled: false });
    const value = values[m[1]]?.trim();
    out.push(value ? { text: value, blank: m[1], filled: true } : { text: m[0], blank: m[1], filled: false });
    last = start + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), blank: null, filled: false });
  return out;
}

export function fill(text: string, values: Record<string, string>): string {
  return segments(text, values).map((s) => s.text).join("");
}

/** "client_name" -> "Client name" for the backend's missing_fields. */
export function humanize(field: string): string {
  const s = field.replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// Splits the analysis text (rendered by backend/agents/analysis.py `render`) into sections.
// Format: blocks separated by a blank line; a block may start with a "**Heading**" line,
// followed by "- item" bullets, "1. item" steps, or plain text.

export interface Section {
  heading: string | null;
  kind: "bullets" | "numbered" | "text";
  items: string[];
}

// Sections the UI shows as separate cards from the structured fields
// (missing_facts / uncertainties / questions_for_lawyer), so they are not repeated in the text.
// Must match HEADINGS in backend/agents/analysis.py.
export const CARD_HEADINGS = new Set([
  "Missing facts", "Uncertainties", "Questions for the lawyer",
  "अनुपलब्ध तथ्य", "अनिश्चितताएँ", "वकील के लिए प्रश्न",
]);

const HEADING = /^\*\*(.+?)\*\*$/;
const BULLET = /^[-*]\s+/;
const NUMBERED = /^\d+[.)]\s+/;

export function parseAnswer(text: string): Section[] {
  const sections: Section[] = [];
  for (const block of text.replace(/\r\n/g, "\n").split(/\n\s*\n/)) {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    let heading: string | null = null;
    const m = lines[0].match(HEADING);
    if (m) {
      heading = m[1].trim();
      lines.shift();
    }
    if (lines.length && lines.every((l) => BULLET.test(l))) {
      sections.push({ heading, kind: "bullets", items: lines.map((l) => l.replace(BULLET, "")) });
    } else if (lines.length && lines.every((l) => NUMBERED.test(l))) {
      sections.push({ heading, kind: "numbered", items: lines.map((l) => l.replace(NUMBERED, "")) });
    } else {
      sections.push({ heading, kind: "text", items: lines.length ? [lines.join(" ")] : [] });
    }
  }
  return sections;
}

/** Inline **bold** spans: [text, isBold][] (no HTML is ever injected). */
export function inlineBold(text: string): Array<[string, boolean]> {
  const out: Array<[string, boolean]> = [];
  text.split(/\*\*(.+?)\*\*/g).forEach((part, i) => {
    if (part) out.push([part, i % 2 === 1]);
  });
  return out;
}

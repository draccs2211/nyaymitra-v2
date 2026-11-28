import { CARD_HEADINGS, inlineBold, parseAnswer } from "../lib/answer";

// Same shape as backend/agents/analysis.py render()
const RENDERED = [
  "**Summary**\nTwo months of unpaid wages.",
  "**Applicable law (verify before citing)**\n- Code on Wages, 2019 s.17\n- Code on Wages, 2019 s.45",
  "**Recommended steps**\n1. Collect salary slips\n2. Send a legal notice",
  "**Missing facts**\n- Monthly wage",
  "**Timeline**\nA few weeks.",
].join("\n\n");

describe("parseAnswer", () => {
  it("splits headings, bullets, numbered steps and text", () => {
    const s = parseAnswer(RENDERED);
    expect(s.map((x) => [x.heading, x.kind])).toEqual([
      ["Summary", "text"],
      ["Applicable law (verify before citing)", "bullets"],
      ["Recommended steps", "numbered"],
      ["Missing facts", "bullets"],
      ["Timeline", "text"],
    ]);
    expect(s[1].items).toEqual(["Code on Wages, 2019 s.17", "Code on Wages, 2019 s.45"]);
    expect(s[2].items).toEqual(["Collect salary slips", "Send a legal notice"]);
  });

  it("keeps text without a heading (e.g. emergency or fallback responses)", () => {
    expect(parseAnswer("Call 112 now.\nThen contact DLSA.")).toEqual([
      { heading: null, kind: "text", items: ["Call 112 now. Then contact DLSA."] },
    ]);
  });

  it("knows the card headings in both languages", () => {
    for (const h of ["Missing facts", "Uncertainties", "Questions for the lawyer", "अनुपलब्ध तथ्य"]) {
      expect(CARD_HEADINGS.has(h)).toBe(true);
    }
    expect(CARD_HEADINGS.has("Summary")).toBe(false);
  });
});

describe("inlineBold", () => {
  it("marks bold runs without producing HTML", () => {
    expect(inlineBold("see **BNS 318** and <b>x</b>")).toEqual([
      ["see ", false], ["BNS 318", true], [" and <b>x</b>", false],
    ]);
  });
});

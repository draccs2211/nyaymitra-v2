import { blanks, fill, humanize, segments } from "../lib/draft";

const TEXT = "To,\n[Employer Name]\nPay Rs [Amount] for [Period]. Regards, [Employer Name]";

describe("draft blanks", () => {
  it("lists unique placeholders in order", () => {
    expect(blanks(TEXT)).toEqual(["Employer Name", "Amount", "Period"]);
  });

  it("fills given values everywhere and leaves the rest as blanks", () => {
    expect(fill(TEXT, { "Employer Name": "ABC Ltd", Amount: " " }))
      .toBe("To,\nABC Ltd\nPay Rs [Amount] for [Period]. Regards, ABC Ltd");
  });

  it("marks filled and unfilled segments", () => {
    const s = segments("A [X] B [Y]", { X: "1" });
    expect(s.filter((x) => x.blank).map((x) => [x.text, x.filled])).toEqual([["1", true], ["[Y]", false]]);
    expect(s.map((x) => x.text).join("")).toBe("A 1 B [Y]");
  });

  it("works with Hindi placeholders", () => {
    expect(blanks("सेवा में, [नियोक्ता का नाम]")).toEqual(["नियोक्ता का नाम"]);
  });

  it("humanizes backend field names", () => {
    expect(humanize("legal_notice")).toBe("Legal notice");
  });
});

import { fireEvent, render, screen } from "@testing-library/react";
import { DraftPanel } from "../components/DraftPanel";
import { Progress, stageIndex } from "../components/Progress";
import { canDecide } from "../lib/auth";
import type { Me } from "../lib/api";

describe("Progress", () => {
  it("orders stages as the pipeline runs them", () => {
    expect(stageIndex("screening")).toBe(0);
    expect(stageIndex("researching")).toBe(2);
    expect(stageIndex("finding_legal_aid")).toBe(stageIndex("verifying"));
    expect(stageIndex("completed")).toBe(6);
  });

  it("marks the running stage", () => {
    render(<Progress stage="analysing" />);
    expect(screen.getByText("Analysis").closest("li")?.getAttribute("aria-current")).toBe("step");
  });
});

describe("DraftPanel", () => {
  it("fills blanks live and counts what is left", () => {
    render(<DraftPanel english={"To [Employer Name], pay [Amount]."} hindi={null} />);
    expect(screen.getByText("2 blanks to fill")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Employer Name"), { target: { value: "ABC Ltd" } });
    expect(screen.getByText("ABC Ltd").tagName).toBe("MARK");
    expect(screen.getByText("1 blank to fill")).toBeTruthy();
  });

  it("is read-only in review", () => {
    render(<DraftPanel english={"To [Employer Name]"} hindi={"सेवा में [नाम]"} editable={false} />);
    expect(screen.queryByRole("textbox")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "हिंदी" }));
    expect(screen.getByText("[नाम]")).toBeTruthy();
  });
});

describe("canDecide (mirrors backend can_review)", () => {
  const me = (role: Me["role"], firm_id: string | null): Me => ({ id: "u1", name: "A", email: "a@b.in", role, firm_id });
  it("firm seniors and admins decide; associates do not", () => {
    expect(canDecide(me("senior", "f"), "u2")).toBe(true);
    expect(canDecide(me("admin", "f"), "u2")).toBe(true);
    expect(canDecide(me("associate", "f"), "u1")).toBe(false);
  });
  it("solo users decide only their own", () => {
    expect(canDecide(me("associate", null), "u1")).toBe(true);
    expect(canDecide(me("associate", null), "u2")).toBe(false);
  });
});

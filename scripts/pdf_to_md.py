"""Convert an India Code Act PDF into the section-wise .md format used by data/legal_corpus.

Usage: python scripts/pdf_to_md.py <pdf> <out.md> --act "..." --domain criminal --source-url URL --date 2023-12-25 [--jurisdiction IN]
Prints a validation report: sections parsed vs sections listed in the Act's own arrangement-of-sections.
"""
import argparse, re, subprocess, sys

SEC = re.compile(r"^\s{0,12}(\d{1,3}[A-Z]{0,2})\.\s+(.{3,250}?)\.\s?[—–-]\s*(.*)$")
ARR = re.compile(r"^\s{0,12}(\d{1,3}[A-Z]{0,2})\.\s+([A-Z].{2,250})$")
SCHED = re.compile(r"^\s*(THE\s+)?(FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE\b")
NOISE = {"e", "od", "ac", "di", "in", "dia", "ndia", "code", "i", "n", "d", "a", "c", "o"}

def key(n):
    m = re.match(r"(\d+)([A-Z]*)", n)
    return int(m.group(1)), m.group(2)

def clean_lines(text):
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if s.lower() in NOISE or re.fullmatch(r"\d{1,3}", s):  # watermark bits, page numbers
            continue
        out.append(re.sub(r"\s+(od|aC|di)$", "", ln.rstrip()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out")
    ap.add_argument("--act", required=True); ap.add_argument("--domain", required=True)
    ap.add_argument("--source-url", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--jurisdiction", default="IN")
    ap.add_argument("--source-name", default="India Code (official text)")
    a = ap.parse_args()

    raw = subprocess.run(["pdftotext", "-layout", a.pdf, "-"], capture_output=True, text=True, check=True).stdout
    lines = clean_lines(raw)
    # join section headings whose title wraps onto the next line
    WRAP2 = re.compile(r"^[^\n]{0,250}?\.\s?[\u2014\u2013-]")
    merged, i = [], 0
    while i < len(lines):
        ln = lines[i]
        j = i + 1
        while j < len(lines) and not lines[j].strip() and j - i < 8:
            j += 1
        nxt = lines[j] if j < len(lines) else ""
        if (ARR.match(ln) and not SEC.match(ln) and nxt.strip() and not re.match(r"^\s{0,12}\d{1,3}[A-Z]{0,2}\.\s", nxt)
                and WRAP2.match(nxt.strip()) and SEC.match(ln.rstrip() + " " + nxt.strip())):
            merged.append(ln.rstrip() + " " + nxt.strip())
            i = j + 1
        else:
            merged.append(ln)
            i += 1
    lines = merged

    # expected sections = arrangement list (appears before the first body section)
    body_start = next((i for i, ln in enumerate(lines) if (m := SEC.match(ln)) and m.group(1) == "1"), 0)
    expected = [m.group(1) for ln in lines[:body_start] if (m := ARR.match(ln))]
    # sections in order
    secs, cur, last = [], None, (0, "")
    for ln in lines[body_start:]:
        m = SEC.match(ln)
        if m and key(m.group(1)) > last and key(m.group(1))[0] - last[0] <= 5:
            last = key(m.group(1))
            cur = {"num": m.group(1), "title": m.group(2).strip(), "body": [m.group(3)]}
            secs.append(cur)
        elif cur is not None:
            if SCHED.match(ln) and expected and key(cur["num"]) >= key(expected[-1]):
                break
            cur["body"].append(ln.strip())

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"---\nact: {a.act}\ndomain: {a.domain}\njurisdiction: {a.jurisdiction}\n"
                f"source_url: {a.source_url}\nsource_name: {a.source_name}\ndate: {a.date}\n---\n\n")
        for s in secs:
            body = re.sub(r"\n{3,}", "\n\n", "\n".join(s["body"])).strip()
            body = re.sub(r"(?<=[a-z,;])\n(?=[a-z(])", " ", body)
            f.write(f"## Section {s['num']} — {s['title']}\n{body}\n\n")

    got = {s["num"] for s in secs}
    missing = [e for e in expected if e not in got]
    extra = [g for g in got if g not in set(expected)]
    short = [s["num"] for s in secs if len(" ".join(s["body"])) < 40]
    print(f"{a.pdf}\n  expected(arrangement)={len(expected)} parsed={len(secs)} missing={missing[:40]} extra={sorted(extra)[:20]} very_short={short[:20]}")

if __name__ == "__main__":
    sys.exit(main())

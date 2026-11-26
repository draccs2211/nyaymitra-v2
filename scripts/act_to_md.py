"""Flexible Act PDF -> section .md using the Act's own arrangement-of-sections as the expected list.
--style dash   : body headings look like "12. Title.—text" (India Code)
--style margin : body is "12. text ...        Marginal title" (right-hand margin note; title from arrangement)
Prints a validation report (expected vs parsed vs missing)."""
import argparse, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("out")
ap.add_argument("--act", required=True); ap.add_argument("--domain", required=True)
ap.add_argument("--source-url", required=True); ap.add_argument("--date", required=True)
ap.add_argument("--jurisdiction", default="IN"); ap.add_argument("--style", default="dash", choices=["dash", "margin"])
ap.add_argument("--source-name", default="India Code (official text)")
a = ap.parse_args()

raw = subprocess.run(["pdftotext", "-layout", a.pdf, "-"], capture_output=True, text=True, check=True).stdout
NOISE = {"e", "od", "ac", "di", "in", "dia", "ndia", "code", "i", "n", "d", "a", "c", "o"}
L = []
for ln in raw.splitlines():
    s = ln.strip()
    if s.lower() in NOISE or re.fullmatch(r"\d{1,3}", s):
        continue
    L.append(re.sub(r"\s+(od|aC|di)$", "", ln.rstrip()))

NUM = r"\d{1,3}[A-Z]{0,2}"
# body start: section 1 line (not in the index) mentioning short title / may be called
body = None
for i, l in enumerate(L):
    if i > 25 and re.match(rf"^\s{{0,16}}\[?1\.\s", l) and re.search(r"Short title|may be called|short title", " ".join(L[i:i + 3])) and "—" in " ".join(L[i:i + 3]) or \
       (i > 25 and re.match(r"^\s{0,16}\[?1\.\s+\(1\)", l)):
        body = i; break
if body is None: sys.exit("could not find body start")

ARR = re.compile(rf"^\s{{0,12}}\[?({NUM})\.\s+(\S.*)$")
ent = []
for l in L[:body]:
    m = ARR.match(l); s = l.strip()
    if m and not re.search(r"\(\d+ of \d{4}\)", s): ent.append([m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()])
    elif ent and s and not s.isupper() and not s.startswith(("CHAPTER", "SECTION", "THE ", "ARRANGEMENT", "STATEMENT")) and not ent[-1][1].endswith("."):
        ent[-1][1] += " " + s
# collapse duplicates (keep first occurrence, ordered)
seen, E = set(), []
for n, t in ent:
    if n not in seen: seen.add(n); E.append([n, t])
ent = E

if not ent and a.style == "margin":      # Acts without an arrangement list: walk numbers 1,1A?,2,... in the body
    cur, p0 = 0, body
    while True:
        found = None
        for cand in (f"{cur+1}",):
            for i in range(p0, len(L)):
                if re.match(rf"^\s{{0,16}}{cand}\.\s+\S.*\s{{6,}}\S", L[i]) or re.match(rf"^\s{{0,16}}{cand}\.\s+\(\d\)", L[i]):
                    found = (cand, i); break
            if found: break
        if not found: break
        ent.append([found[0], ""]); cur += 1; p0 = found[1] + 1

START = lambda n: re.compile(rf"^\s{{0,16}}\[?\s?{re.escape(n)}\s?\.(\s|(?=[(A-Z\"“\[]))")
pos, p = [], body
for n, t in ent:
    cand = None
    for i in range(p, len(L)):
        if START(n).match(L[i]):
            ctx = " ".join(L[i:i + 2])
            if a.style == "margin" or re.search(r"[—–]", ctx[:300]) or re.search(r"Omitted|Repealed|Rep\.", ctx[:200]):
                cand = i; break
    pos.append(cand)
    if cand is not None: p = cand + 1

valid = [(e, q) for e, q in zip(ent, pos) if q is not None]
missing = [e[0] for e, q in zip(ent, pos) if q is None]

def cut_margin(l):
    if re.match(r"^\s{40,}\S", l): return ""
    return re.split(r"\s{8,}", l.rstrip())[0] if a.style == "margin" else l

secs = []
for k, (e, q) in enumerate(valid):
    end = valid[k + 1][1] if k + 1 < len(valid) else len(L)
    ch = L[q:end]
    if k + 1 == len(valid):
        for j, l in enumerate(ch):
            if re.match(r"^\s*(THE\s+)?(FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE\b", l.strip()): ch = ch[:j]; break
    ch = [cut_margin(l) for l in ch]
    if a.style == "margin" and not e[1]:
        frags = []
        for l in L[q:end]:
            parts = re.split(r"\s{8,}", l.strip(), maxsplit=1) if re.match(r"^\s{0,16}\S", l) and not re.match(r"^\s{40,}", l) else [l.strip()]
            frag = parts[1].strip() if len(parts) == 2 else (parts[0] if re.match(r"^\s{40,}\S", l) else "")
            if frag: frags.append(frag)
            if frag.endswith(".") and len(frags) >= 1: break
        e = [e[0], " ".join(frags)]
    first = re.sub(rf"^\s*\[?\s?{re.escape(e[0])}\s?\.\s*", "", ch[0])
    title = e[1].rstrip(".")
    if a.style == "dash" and not first.startswith("("):
        RX = r"^\s*(.{3,250}?)\.\s?[\u2014\u2013]\s*(.*)$"
        m1 = re.match(RX, first)
        if m1:
            title, first = m1.group(1).strip().lstrip("["), m1.group(2)
        elif len(ch) > 1:
            m2 = re.match(RX, first + " " + ch[1].strip())
            if m2 and len(m2.group(1)) > len(first):       # title wrapped onto next line
                title, first = m2.group(1).strip().lstrip("["), m2.group(2)
                ch = [ch[0], ""] + ch[2:]
    body_lines = [first] + [l.strip() for l in ch[1:]]
    body_lines = [l for l in body_lines if not re.fullmatch(r"\d{1,3}", l)]
    txt = re.sub(r"\n{2,}", "\n", "\n".join(body_lines)).strip()
    txt = re.sub(r"(?<=[a-z,;])\n(?=[a-z(])", " ", txt)
    secs.append((e[0], title, txt))

with open(a.out, "w", encoding="utf-8") as f:
    f.write(f"---\nact: {a.act}\ndomain: {a.domain}\njurisdiction: {a.jurisdiction}\nsource_url: {a.source_url}\nsource_name: {a.source_name}\ndate: {a.date}\n---\n\n")
    for n, t, b in secs: f.write(f"## Section {n} — {t}\n{b}\n\n")
short = [n for n, t, b in secs if len(b) < 25]
print(f"{a.pdf.split('/')[-1]}: expected={len(ent)} parsed={len(secs)} missing={missing} very_short={short}")

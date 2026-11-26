"""Gazette of India Act PDF (as enacted) -> section-wise .md for data/legal_corpus.

Gazette prints have no arrangement-of-sections; each section title is a margin note (left
margin on some pages, right on others) mixed with citation notes such as "14 of 1947.".
Uses word coordinates (poppler `pdftotext -bbox`, NOT xpdf) to separate body from margin:
per page, the body's left/right edges are the most common line start/end x; words outside
them are margin. Sections are matched strictly in order 1, 2, 3, ... so a miss shows up as a
count mismatch in the report instead of a silently wrong file.

Usage:
  python scripts/gazette_to_md.py <pdf> <out.md> --act "..." --domain labor --source-url URL
      --date YYYY-MM-DD --expected N [--source-name ...] [--jurisdiction IN] [--pdftotext PATH]
PDFTOTEXT env var can also point at poppler's pdftotext. Schedules are not included.
"""
import argparse, collections, html, os, re, subprocess, sys

HEADER = re.compile(r"GAZETTE OF INDIA|^SEC\.\s*1\]|^\[?\s*PART\s+II|^\d{1,3}$|MGIPMRND|UPLOADED BY|PUBLISHED BY THE CONTROLLER")
CITATION = re.compile(r"^\d{1,3}\s*of\s*\d{4}\.?$")
CAPS = re.compile(r"^[A-Z][A-Z ,.'\-–—()]{3,}$")
WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')


def pages(pdf, exe):
    out = subprocess.run([exe, "-bbox", pdf, "-"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=True).stdout
    for chunk in out.split("<page ")[1:]:
        yield [(float(a), float(b), float(c), float(d), html.unescape(t)) for a, b, c, d, t in WORD.findall(chunk)]


def group_lines(words, tol=3.0):
    lines = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if lines and abs(lines[-1][0] - w[1]) <= tol:
            lines[-1][1].append(w)
        else:
            lines.append([w[1], [w]])
    return [(y, sorted(ws, key=lambda w: w[0])) for y, ws in lines]


def mode(values, step=3):
    c = collections.Counter(round(v / step) * step for v in values)
    return c.most_common(1)[0][0] if c else 0


def clusters(ws, gap=14.0):
    """Split a line's words where the horizontal gap is larger than normal word spacing."""
    out = [[ws[0]]]
    for w in ws[1:]:
        if w[0] - out[-1][-1][2] > gap:
            out.append([w])
        else:
            out[-1].append(w)
    return out


def gutter(words, lo, hi):
    """Widest x-strip in [lo, hi] crossed by no word but with words on both sides -> (x0, x1) or None."""
    cov = [False] * 700
    for w in words:
        for x in range(max(0, int(w[0])), min(699, int(w[2]) + 1)):
            cov[x] = True
    best, x = None, lo
    while x < hi:
        if not cov[x]:
            s = x
            while x < hi and not cov[x]:
                x += 1
            if any(cov[lo - 40:s]) and any(cov[x:x + 40]) and (best is None or x - s > best[1] - best[0]):
                best = (s, x)
        x += 1
    return best if best and best[1] - best[0] >= 4 else None


def page_rows(words):
    """-> (body_lines [(y, indent, text)], margin_lines [(y, text)])"""
    core = [w for w in words if 85 < w[1] < 800]          # skip running header / footer band
    if not core:
        return [], []
    gl = gutter(core, 40, 230)                              # left margin | body
    gr = gutter(core, 380, 580)                             # body | right margin
    left_edge = gl[1] if gl else 0
    right_edge = gr[0] if gr else 9999
    body, margin = [], []
    for y, ws in group_lines(core):
        b = [w for w in ws if w[0] >= left_edge - 1 and w[2] <= right_edge + 1]
        lm = [w for w in ws if w[2] < left_edge - 1]
        rm = [w for w in ws if w[0] > right_edge + 1]
        if b:
            t = " ".join(w[4] for w in b)
            if not HEADER.search(t):
                body.append((y, b[0][0] - (left_edge or min(w[0] for w in core)), t))
        for side in (lm, rm):
            if side:
                margin.append((y, " ".join(w[4] for w in side)))
    return body, margin


def join_frags(frags):
    out = ""
    for f in frags:
        out = out + f if out.endswith("-") else (out + " " + f if out else f)
    return re.sub(r"\s+", " ", out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out")
    ap.add_argument("--act", required=True); ap.add_argument("--domain", required=True)
    ap.add_argument("--source-url", required=True); ap.add_argument("--date", required=True)
    ap.add_argument("--jurisdiction", default="IN")
    ap.add_argument("--source-name", default="Gazette of India (official text)")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--pdftotext", default=os.environ.get("PDFTOTEXT", "pdftotext"))
    a = ap.parse_args()

    rows = []          # (page, y, indent, body_text)
    margins = []       # (page, y, text)
    for p, words in enumerate(pages(a.pdf, a.pdftotext)):
        body, margin = page_rows(words)
        rows += [(p, y, ind, t) for y, ind, t in body]
        margins += [(p, y, t) for y, t in margin if not CITATION.match(t) and not HEADER.search(t)]

    first = next((i for i, r in enumerate(rows) if re.match(r"^1\s?[.,]\s*\(\S{1,2}\)\s*This\s+(Act|Code)", r[3])), None)
    if first is None:
        sys.exit("could not find section 1")
    rows = rows[first:]
    stop = next((i for i, r in enumerate(rows) if re.match(r"^THE\s+(FIRST\s+)?SCHEDULE\b", r[3])), len(rows))
    rows = rows[:stop]

    starts, n = [], 1
    for i, r in enumerate(rows):
        if re.match(rf"^{n}\s?[.,:]\s*(\(|[A-Z“\"‘'])", r[3]):
            starts.append((n, i)); n += 1

    secs = []
    for k, (num, i) in enumerate(starts):
        j = starts[k + 1][1] if k + 1 < len(starts) else len(rows)
        page, y0 = rows[i][0], rows[i][1]
        # title: margin lines on the same page from the section's first line downward
        frags = []
        for mp, my, mt in margins:
            if mp == page and y0 - 4 <= my <= y0 + 90:
                frags.append(mt)
                if mt.endswith("."):
                    break
        title = join_frags(frags).rstrip(".") or "(title not found in margin)"
        paras, cur = [], []
        for idx in range(i, j):
            _, _, ind, t = rows[idx]
            if idx == i:
                t = re.sub(rf"^{num}\s?[.,:]\s*", "", t)
            if CAPS.match(t):
                continue
            if cur and (ind > 8 or t.startswith("(")):
                paras.append(join_frags(cur)); cur = []
            cur.append(t)
        if cur:
            paras.append(join_frags(cur))
        secs.append((str(num), title, "\n".join(paras)))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"---\nact: {a.act}\ndomain: {a.domain}\njurisdiction: {a.jurisdiction}\n"
                f"source_url: {a.source_url}\nsource_name: {a.source_name}\ndate: {a.date}\n---\n\n")
        for num, t, b in secs:
            f.write(f"## Section {num} — {t}\n{b}\n\n")
    untitled = [s for s, t, _ in secs if t.startswith("(title")]
    short = [s for s, _, b in secs if len(b) < 40]
    ok = "OK" if not a.expected or len(secs) == a.expected else f"MISMATCH (expected {a.expected})"
    print(f"{os.path.basename(a.pdf)}: parsed={len(secs)} {ok} untitled={untitled} short={short}")


if __name__ == "__main__":
    main()

"""RERA 2016 gazette PDF (no arrangement list): sequential sections 1..N from raw text; margin note = title."""
import re, subprocess, sys
pdf, out = sys.argv[1], sys.argv[2]
raw = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True).stdout
L = [l.rstrip() for l in raw.splitlines()]
HDR = re.compile(r"^(GMGIPMRND.*|THE GAZETTE OF INDIA EXTRAORDINARY|\[?P ?ART II.*|SEC\. ?1\]|No\. ?\d+\]|\d{1,3})$|^\[?PART II")
body = next(i for i, l in enumerate(L) if re.match(r"^1\.\s+\(1\)\s+This Act", l))
pos, p, n = [], body, 1
while True:
    q = next((i for i in range(p, len(L)) if re.match(rf"^{n}\.\s+\S", L[i]) and (re.match(rf"^{n}\.\s+\(\d\)", L[i]) or len(L[i]) > 30)), None)
    if q is None: break
    pos.append((n, q)); p = q + 1; n += 1
secs = []
for k, (num, q) in enumerate(pos):
    end = pos[k + 1][1] if k + 1 < len(pos) else len(L)
    ch = [re.sub(rf"^{num}\.\s*", "", L[q])] + L[q + 1:end]
    if k + 1 == len(pos):
        for j, l in enumerate(ch):
            if re.match(r"^(THE\s+)?SCHEDULE", l.strip()): ch = ch[:j]; break
    blocks, cur = [], []
    for l in ch + [""]:
        s = l.strip()
        if s and not HDR.match(s): cur.append(s)
        elif not s and cur: blocks.append(cur); cur = []
    title = None; keep = []
    for bi, b in enumerate(blocks):
        t = " ".join(b)
        if title is None and bi > 0 and len(t) < 95 and t[:1].isupper() and t.endswith(".") and not t.startswith("(") and not re.match(r"^(Explanation|Provided|Illustration)", t) and len(b) <= 3:
            title = t.rstrip("."); continue
        if re.match(r"^(CHAPTER [IVX]+|[A-Z ,\-]{6,})$", t): continue
        keep.append(" ".join(b))
    bodytxt = "\n".join(keep)
    if not title:
        first = re.sub(r"^\(\d+\)\s*", "", bodytxt.strip())
        title = "(untitled in source) " + " ".join(first.split()[:9]).rstrip(",;:—") + "..."
    secs.append((str(num), title, bodytxt))
with open(out, "w", encoding="utf-8") as f:
    f.write("---\nact: The Real Estate (Regulation and Development) Act, 2016\ndomain: property\njurisdiction: IN\nsource_url: https://www.indiacode.nic.in\nsource_name: India Code (official text)\ndate: 2016-03-26\n---\n\n")
    for n, t, b in secs: f.write(f"## Section {n} — {t}\n{b}\n\n")
notitle = [n for n, t, b in secs if t.startswith("(untitled")]
print(f"parsed={len(secs)} untitled={len(notitle)} short={[n for n,t,b in secs if len(b)<40]}")

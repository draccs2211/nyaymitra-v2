"""Cross-check hard-coded IPC/BNS references against data/reference/ipc_bns_mapping.json and the corpus text."""
import json, re, ast, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
m = json.load(open(root / "data/reference/ipc_bns_mapping.json"))
def titles(f):
    return dict(re.findall(r"^## Section (\S+) — (.*)$", (root / "data/legal_corpus" / f).read_text(encoding="utf-8"), re.M))
bns_t, bnss_t = titles("bns_2023.md"), titles("bnss_2023.md")
issues = []
issues = issues
# 2) free-text "Section X IPC / Section Y BNS"
print("== free-text references")
for f in [*root.glob("backend/**/*.py"), root / "data/legal_corpus/curated_summaries.jsonl"]:
    for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        for mm in re.finditer(r"Section (\d+[A-Z]?) IPC\s*/\s*Section (\d+[A-Z]?) BNS(?!S)", line):
            ipc, bns = mm.groups()
            right = m["ipc_to_bns"].get(ipc)
            ok = right is not None and bns in right
            print(f"  {f.relative_to(root)}:{ln}  IPC {ipc} / BNS {bns}  table={right}  {'OK' if ok else '<<< MISMATCH'}")
            if not ok: issues.append((str(f.relative_to(root)), ipc, bns, right))
# 3) BNSS titles for CrPC refs used in code
print("== BNSS sections cited (check titles)")
for s in ("173", "482", "480", "483", "35"):
    print(f"  BNSS {s}: {bnss_t.get(s)}")
print("ISSUES:", issues)

# Legal corpus

Everything NyayMitra can retrieve and cite comes from the files in this folder.
After editing, run:

```powershell
python -m backend.data_loader --dry-run   # validate only
python -m backend.data_loader             # load (idempotent; removes docs no longer in these files)
```

**Only add text a lawyer has checked.** Analysis and drafting may cite only provisions
that appear here (the citation guard flags anything else).

## 1. Official Act text — `*.md` (preferred)

One file per Act, copied from the official source (India Code: <https://www.indiacode.nic.in>).
The header is required; each `## Section N — Title` heading starts one section.

```markdown
---
act: The Bharatiya Nyaya Sanhita, 2023
domain: criminal
jurisdiction: IN
source_url: https://www.indiacode.nic.in/handle/...      # exact page of this Act
source_name: India Code (official text)
date: 2023-12-25
---

## Section 316 — Criminal breach of trust
(1) Whoever, being in any manner entrusted with property, ...

(2) ...

## Section 318 — Cheating
...
```

- `domain`: one of `criminal, civil, family, property, consumer, rti, labor, constitutional, general`
- `jurisdiction`: `IN` for central Acts, `IN-UP`, `IN-DL`, ... for state Acts
- `date`: `YYYY`, `YYYY-MM` or `YYYY-MM-DD`
- Headings may be `Section`, `Sec.` or `धारा`. Sections longer than ~1500 characters are split into
  overlapping parts automatically.
- `source_url` is mandatory: official text without a source is rejected.

## 2. Curated documents — `*.jsonl`

One JSON object per line: `{"content": "...", "metadata": {...}}`. Used for practical guidance,
helplines and short summaries. `curated_summaries.jsonl` holds the original 26 hand-written
summaries — **these are unreviewed** and marked as such in `source_name`.

Metadata fields (validated by `backend/rag/metadata.py`):

| Field | Required | Notes |
|---|---|---|
| `doc_type` | yes | `act`, `legal_guidance`, `helplines`, `ipc_section`, `case_law`, `statute_section` |
| `domain` | yes | see list above |
| `act`, `section` | for `act` / `ipc_section` / `statute_section` | official short title, e.g. `The Consumer Protection Act, 2019` |
| `court`, `date`, `source_url` | for `case_law` | never add a judgment without its source |
| `jurisdiction` | no (default `IN`) | `IN` or `IN-XX` |
| `source_name`, `topic` | no | |

Unknown keys, invalid values, duplicate text and missing required fields fail validation
with the file and line number.

"""
PAID: calls the real orchestrator (gpt-4o-mini) once per eval query to collect
its English search rewrite (search_query_en) and statutory terms (search_terms_en),
for the live retrieval eval.

    python -m backend.evaluation.generate_rewrites --out rewrites.json [--langs hi,hinglish]
    python -m backend.evaluation.retrieval_eval --llm-rewrites rewrites.json

18 items x languages requested = number of OpenAI calls (54 for all three).
"""
import argparse
import json

from backend.evaluation.retrieval_eval import LANGS, load_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--langs", default=",".join(LANGS))
    args = parser.parse_args()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    from backend.agents.orchestrator import orchestrator_agent
    from backend.state import initial_state

    rewrites = {}
    for item in load_items():
        for lang in langs:
            key = f"{item['id']}:{lang}"
            out = orchestrator_agent(initial_state(user_query=item[lang]))
            rewrites[key] = [out.get("search_query_en", ""), out.get("search_terms_en", "")]
            print(f"{key:<32} -> {rewrites[key]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rewrites, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(rewrites)} rewrites to {args.out}")


if __name__ == "__main__":
    main()

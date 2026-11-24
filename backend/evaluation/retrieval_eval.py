"""
Retrieval evaluation — Hit@1, Hit@5, MRR@10 per language and retrieval method.
Local only (no OpenAI; Cohere rerank disabled).

Usage:
    python -m backend.evaluation.retrieval_eval            # baseline methods
    python -m backend.evaluation.retrieval_eval --json out.json
"""
import argparse
import json
import os
from typing import Callable, Dict, List

EVAL_SET = os.path.join(os.path.dirname(__file__), "retrieval_eval_set.json")
LANGS = ("en", "hi", "hinglish")
K = 10


def is_relevant(meta: dict, rules: List[dict]) -> bool:
    for rule in rules:
        ok = True
        for field, value in rule.items():
            have = str(meta.get(field, ""))
            if field == "act":
                ok = ok and value.lower() in have.lower()
            else:
                ok = ok and have.upper() == str(value).upper()
        if ok:
            return True
    return False


def score(ranked_metas: List[dict], rules: List[dict]) -> Dict[str, float]:
    rank = next((i + 1 for i, m in enumerate(ranked_metas[:K]) if is_relevant(m, rules)), None)
    return {
        "hit@1": float(rank == 1),
        "hit@5": float(rank is not None and rank <= 5),
        "mrr@10": 1.0 / rank if rank else 0.0,
    }


def load_items() -> List[dict]:
    with open(EVAL_SET, encoding="utf-8") as f:
        return json.load(f)["items"]


def evaluate(methods: Dict[str, Callable[[dict, str], List[dict]]], items: List[dict]) -> Dict:
    """methods: name -> fn(item, lang) returning ranked docs with 'metadata'."""
    results: Dict = {}
    misses: Dict = {}
    for name, fn in methods.items():
        results[name] = {}
        for lang in LANGS:
            rows = []
            for item in items:
                docs = fn(item, lang)
                s = score([d.get("metadata") or {} for d in docs], item["relevant"])
                rows.append(s)
                if not s["hit@5"]:
                    misses.setdefault(name, {}).setdefault(lang, []).append(item["id"])
            results[name][lang] = {
                m: round(sum(r[m] for r in rows) / len(rows), 3) for m in ("hit@1", "hit@5", "mrr@10")
            }
    return {"n_items": len(items), "results": results, "misses@5": misses}


def print_table(report: Dict) -> None:
    print(f"\nItems per language: {report['n_items']}")
    header = f"{'method':<22}" + "".join(f"{lang + ' ' + m:>16}" for lang in LANGS for m in ("hit@5", "mrr@10"))
    print(header)
    print("-" * len(header))
    for name, by_lang in report["results"].items():
        row = f"{name:<22}"
        for lang in LANGS:
            row += f"{by_lang[lang]['hit@5']:>16.3f}{by_lang[lang]['mrr@10']:>16.3f}"
        print(row)
    print("\nMisses @5:")
    for name, by_lang in report["misses@5"].items():
        for lang, ids in by_lang.items():
            print(f"  {name:<20} {lang:<9} {', '.join(ids)}")


def default_methods(llm_rewrites: Dict[str, str] = None) -> Dict[str, Callable[[dict, str], List[dict]]]:
    from backend.rag.hybrid_search import hybrid_rag
    hybrid_rag.cohere_client = None   # no paid rerank in eval

    methods = {
        "dense": lambda it, lang: hybrid_rag.dense_search(it[lang], K),
        "sparse (BM25)": lambda it, lang: hybrid_rag.sparse_search(it[lang], K),
        "hybrid (RRF)": lambda it, lang: hybrid_rag.retrieve([it[lang]], K),
        # Upper bound for query rewriting: the item's own English phrasing as the rewrite
        "hybrid+en (oracle)": lambda it, lang: hybrid_rag.retrieve([it[lang], it["en"]], K),
    }
    if llm_rewrites:
        methods["hybrid+en (LLM)"] = lambda it, lang: hybrid_rag.retrieve(
            [it[lang], llm_rewrites.get(f"{it['id']}:{lang}", "")], K)
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write full report to this path")
    parser.add_argument("--llm-rewrites", help="JSON {'<item_id>:<lang>': '<english search query>'}")
    args = parser.parse_args()
    llm_rewrites = None
    if args.llm_rewrites:
        with open(args.llm_rewrites, encoding="utf-8") as f:
            llm_rewrites = json.load(f)
    report = evaluate(default_methods(llm_rewrites), load_items())
    print_table(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

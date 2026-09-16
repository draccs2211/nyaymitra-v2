from typing import List, Dict
import json
from datetime import datetime


TEST_CASES = [
    {
        "question": "My landlord is not returning security deposit after 2 months of vacating",
        "expected_domain": "property",
        "expected_keywords": ["deposit", "tenant", "UP Urban Buildings Act", "30 days"]
    },
    {
        "question": "Mere boss ne 3 mahine se salary nahi di",
        "expected_domain": "labor",
        "expected_keywords": ["salary", "Payment of Wages", "labor court"]
    },
    {
        "question": "I need to file RTI for government scheme status",
        "expected_domain": "rti",
        "expected_keywords": ["RTI", "30 days", "Public Information Officer"]
    },
    {
        "question": "Consumer forum complaint against mobile company for defective phone",
        "expected_domain": "consumer",
        "expected_keywords": ["Consumer Protection", "District Commission", "50 lakh"]
    },
    {
        "question": "Police not registering my FIR",
        "expected_domain": "criminal",
        "expected_keywords": ["FIR", "Section 154", "Superintendent of Police", "Zero FIR"]
    }
]


def evaluate_response(
    question: str,
    response: str,
    context: List[str],
    expected_domain: str,
    expected_keywords: List[str]
) -> Dict:
    """Simple evaluation without RAGAS for now."""

    # Check domain
    domain_correct = expected_domain.lower() in response.lower()

    # Check keywords
    keywords_found = [kw for kw in expected_keywords if kw.lower() in response.lower()]
    keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 0

    # Check response length (too short = bad)
    length_score = min(1.0, len(response) / 500)

    # Check disclaimer present
    has_disclaimer = any(word in response.lower() for word in [
        "verify", "consult", "lawyer", "official source", "disclaimer"
    ])

    overall_score = (
        (0.3 * domain_correct) +
        (0.4 * keyword_score) +
        (0.2 * length_score) +
        (0.1 * has_disclaimer)
    )

    return {
        "question": question,
        "domain_correct": domain_correct,
        "keywords_found": keywords_found,
        "keyword_score": round(keyword_score, 2),
        "length_score": round(length_score, 2),
        "has_disclaimer": has_disclaimer,
        "overall_score": round(overall_score, 2)
    }


def run_evaluation(pipeline_results: List[Dict]) -> Dict:
    """Run evaluation on pipeline results."""
    scores = []

    for i, result in enumerate(pipeline_results):
        if i >= len(TEST_CASES):
            break

        test_case = TEST_CASES[i]
        eval_result = evaluate_response(
            question=test_case["question"],
            response=result.get("final_response", ""),
            context=[r.get("content", "") for r in result.get("research_results", [])],
            expected_domain=test_case["expected_domain"],
            expected_keywords=test_case["expected_keywords"]
        )
        scores.append(eval_result)

    avg_score = sum(s["overall_score"] for s in scores) / len(scores) if scores else 0

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_cases": len(scores),
        "average_score": round(avg_score, 2),
        "target_score": 0.75,
        "passed": avg_score >= 0.75,
        "results": scores
    }


def save_eval_results(results: Dict, filepath: str = "evaluation/results.json"):
    try:
        import os
        os.makedirs("evaluation", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {filepath}")
    except Exception as e:
        print(f"Could not save results: {e}")

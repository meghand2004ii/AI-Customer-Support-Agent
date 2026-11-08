import csv
import json
import os

from plan_execute_agent import run_plan_execute

SCENARIO_PATH = "data/raw/agent_scenarios.csv"
OUTPUT_PATH = "outputs/plan_execute_evaluation.json"


def extract_refund_amount(answer):
    import re

    match = re.search(r"REFUND:\s*INR\s*([\d.]+)", answer or "")
    return float(match.group(1)) if match else 0.0


def evaluate_scenarios():
    results = []

    with open(SCENARIO_PATH, "r", encoding="utf-8") as file:
        scenarios = csv.DictReader(file)

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            message = scenario["message"]
            expected = scenario["expected_outcome"]
            expected_refund = float(
                scenario["expected_refund_inr"]
            )

            print("\n" + "=" * 60)
            print("Evaluating:", scenario_id)

            result = run_plan_execute(
                message,
                scenario_id=scenario_id
            )

            actual_status = result["status"]

            decision = result.get("decision", {})
            actual_refund = float(
                decision.get("refund_amount", 0) or 0
            )

            # Compare the actual and expected outcomes.
            if expected == "FALLBACK":
                outcome_match = actual_status.startswith("FALLBACK")
            else:
                outcome_match = actual_status == expected

            if expected in ("REFUND", "ESCALATE"):
                refund_match = abs(
                    actual_refund - expected_refund
                ) < 0.01
            else:
                refund_match = actual_refund == 0.0

            passed = outcome_match and refund_match

            record = {
                "scenario_id": scenario_id,
                "expected_outcome": expected,
                "actual_status": actual_status,
                "expected_refund_inr": expected_refund,
                "actual_refund_inr": actual_refund,
                "outcome_match": outcome_match,
                "refund_match": refund_match,
                "passed": passed,
                "llm_calls": result.get("llm_calls", 0),
                "tokens_used": result.get("tokens_used", 0)
            }

            results.append(record)

            print("Expected:", expected)
            print("Actual:", actual_status)
            print("Passed:", passed)
            print("LLM calls:", record["llm_calls"])
            print("Tokens:", record["tokens_used"])

    total = len(results)
    passed_count = sum(
        1 for result in results if result["passed"]
    )

    evaluation = {
        "total_scenarios": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy_percent": round(
            (passed_count / total) * 100, 2
        ) if total else 0,
        "results": results
    }

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=4)

    print("\n" + "=" * 60)
    print("PLAN-AND-EXECUTE EVALUATION")
    print("Total scenarios:", total)
    print("Passed:", passed_count)
    print("Failed:", total - passed_count)
    print("Accuracy:", evaluation["accuracy_percent"], "%")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate_scenarios()
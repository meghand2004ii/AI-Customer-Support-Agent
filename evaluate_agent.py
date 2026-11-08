import csv
import json
import re

from agent import run_agent
import mock_llm

SCENARIO_FILE = "data/raw/agent_scenarios.csv"
RESULT_FILE = "outputs/agent_evaluation.json"

FINANCIAL_TOOLS = {"issue_refund", "escalate_to_human"}


def evaluate_scenarios():
    results = []

    with open(SCENARIO_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            scenario_id = row["scenario_id"]
            message = row["message"]
            expected = row["expected_outcome"]
            expected_amount = float(row["expected_refund_inr"])

            mock_llm.reset()

            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                max_iterations=6
            )

            actual_answer = result["answer"]
            tools = [step["tool"] for step in result["steps"]]

            # 1. Validate the expected outcome.
            if expected == "REFUND":
                outcome_pass = actual_answer.startswith("REFUND:")
            elif expected == "ESCALATE":
                outcome_pass = actual_answer.startswith("ESCALATE:")
            elif expected == "DENIED":
                outcome_pass = actual_answer.startswith("DENIED:")
            elif expected == "NOT_FOUND":
                outcome_pass = actual_answer.startswith("NOT_FOUND:")
            else:
                outcome_pass = actual_answer.startswith("FALLBACK_")

            # 2. Validate the exact refund amount.
            amount_pass = True

            if expected == "REFUND":
                match = re.search(r"REFUND:\s*INR\s*([\d.]+)", actual_answer)

                amount_pass = (
                    match is not None
                    and abs(float(match.group(1)) - expected_amount) < 0.01
                )

            elif expected == "ESCALATE":
                escalation = next(
                    (
                        step for step in result["steps"]
                        if step["tool"] == "escalate_to_human"
                    ),
                    None
                )

                amount_pass = (
                    escalation is not None
                    and abs(
                        float(escalation["arguments"]["refund_inr"])
                        - expected_amount
                    ) < 0.01
                )

            # 3. Validate expected tool execution.
            expected_tools = {
                "REFUND": [
                    "get_order",
                    "search_policy",
                    "calculate_refund",
                    "issue_refund"
                ],
                "ESCALATE": [
                    "get_order",
                    "search_policy",
                    "calculate_refund",
                    "escalate_to_human"
                ],
                "DENIED": None,
                "NOT_FOUND": ["get_order"],
                "FALLBACK": None
            }

            tool_pass = True

            if expected_tools[expected] is not None:
                tool_pass = tools == expected_tools[expected]

            # 4. Ensure denied and missing orders trigger no financial action.
            safety_pass = True

            if expected in {"DENIED", "NOT_FOUND"}:
                safety_pass = not any(
                    tool in FINANCIAL_TOOLS for tool in tools
                )

            # Final result: all applicable checks must pass.
            passed = (
                outcome_pass
                and amount_pass
                and tool_pass
                and safety_pass
            )

            results.append({
                "scenario_id": scenario_id,
                "expected": expected,
                "expected_refund_inr": expected_amount,
                "actual": actual_answer,
                "status": result["status"],
                "outcome_pass": outcome_pass,
                "amount_pass": amount_pass,
                "tool_pass": tool_pass,
                "safety_pass": safety_pass,
                "passed": passed,
                "tool_steps": len(result["steps"]),
                "tools_executed": tools,
                "total_tokens": result["total_tokens"]
            })

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n========== AGENT EVALUATION ==========\n")

    for result in results:
        print(
            f"{result['scenario_id']} | "
            f"Expected: {result['expected']} | "
            f"Actual: {result['actual']} | "
            f"Outcome: {result['outcome_pass']} | "
            f"Amount: {result['amount_pass']} | "
            f"Tools: {result['tool_pass']} | "
            f"Safety: {result['safety_pass']} | "
            f"Passed: {result['passed']}"
        )

    passed_count = sum(r["passed"] for r in results)
    total_count = len(results)

    accuracy = (
        passed_count / total_count * 100
        if total_count else 0
    )

    print("\n======================================")
    print(f"Total scenarios: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Strict evaluation accuracy: {accuracy:.2f}%")

    if total_count:
        print(
            "Average tokens per scenario:",
            round(sum(r["total_tokens"] for r in results) / total_count, 2)
        )

    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    evaluate_scenarios()
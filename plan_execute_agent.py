import json
import time
import uuid

import mock_llm

from policy_tools import (
    TOOL_SPECS,
    TOOL_FUNCTIONS
)

from agent import SYSTEM_PROMPT


PLAN_EXECUTE_LOG_PATH = "outputs/plan_execute_runs.jsonl"


def create_plan(user_message):
    """
    Ask the LLM planner to generate the complete execution plan
    and record token usage.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a PLAN EXECUTE assistant. "
                "Generate a complete ordered execution plan "
                "for the customer's refund request."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    response = mock_llm.chat(messages)

    plan_response = json.loads(
        response.choices[0].message.content
    )

    return {
        "plan": plan_response.get("plan", []),
        "llm_calls": 1,
        "tokens_used": response.usage.total_tokens
    }

def replan(observation):
    """
    Generate a revised plan and record token usage.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a PLAN EXECUTE assistant. "
                "Revise the execution plan based on the unexpected observation."
            )
        },
        {
            "role": "user",
            "content": f"UNEXPECTED OBSERVATION: {observation}"
        }
    ]

    response = mock_llm.chat(messages)

    revised_response = json.loads(
        response.choices[0].message.content
    )

    return {
        "plan": revised_response.get("plan", []),
        "llm_calls": 1,
        "tokens_used": response.usage.total_tokens
    }

def execute_plan(plan):
    """
    Execute the LLM-generated plan sequentially.
    Stop when an unexpected order result is encountered.
    """

    results = {}

    for step in plan:
        tool_name = step.get("tool")

        if not tool_name:
            continue

        arguments = step.get("arguments", {}).copy()

        print(f"Executing step {step['step']}: {tool_name}")

        tool_function = TOOL_FUNCTIONS.get(tool_name)

        if tool_function is None:
            return {
                "status": "ERROR",
                "error": f"Unknown tool: {tool_name}",
                "results": results
            }

        # Resolve references to previous tool results.
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                reference = value[1:].split(".")
                source = results.get(reference[0])

                if source is None or reference[1] not in source:
                    return {
                        "status": "ERROR",
                        "error": f"Unable to resolve {value}",
                        "results": results
                    }

                arguments[key] = source[reference[1]]

        try:
            result = tool_function(**arguments)
            results[tool_name] = result

            if isinstance(result, dict) and "error" in result:
                return {
                    "status": "UNEXPECTED",
                    "error": result["error"],
                    "results": results
                }

            # Check the order observation before continuing.
            if tool_name == "get_order":

                if result.get("status") != "completed":
                    return {
                        "status": "UNEXPECTED",
                        "error": "Order is not completed.",
                        "results": results
                    }

        except Exception as error:
            return {
                "status": "UNEXPECTED",
                "error": str(error),
                "results": results
            }

    return {
        "status": "COMPLETED",
        "results": results
    }

def decide_next_action(results):
    """
    Decide and execute the appropriate financial action.
    """

    from policy_tools import issue_refund, escalate_to_human

    order = results.get("get_order")
    refund = results.get("calculate_refund")

    if not order or "error" in order:
        return {
            "action": "NOT_FOUND",
            "reason": "Order not found."
        }

    if order.get("status") != "completed":
        return {
            "action": "DENIED",
            "reason": "Order is not completed."
        }

    if not refund:
        return {
            "action": "DENIED",
            "reason": "Refund calculation unavailable."
        }

    refund_amount = refund.get("refund_inr", 0)
    order_id = order["order_id"]

    if refund_amount <= 0:
        return {
            "action": "DENIED",
            "reason": "Order does not qualify for a refund."
        }

    if refund_amount > 1000:
        action_result = escalate_to_human(
            order_id=order_id,
            refund_inr=refund_amount
        )

        return {
            "action": "ESCALATE",
            "refund_amount": refund_amount,
            "reason": "Refund requires human approval.",
            "action_result": action_result
        }

    action_result = issue_refund(
        order_id=order_id,
        refund_inr=refund_amount
    )

    return {
        "action": "REFUND",
        "refund_amount": refund_amount,
        "reason": "Refund successfully recorded.",
        "action_result": action_result
    }

def run_plan_execute(user_message, scenario_id="MANUAL"):
    """
    Run Plan-and-Execute and track LLM calls and tokens.
    """

    run_id = str(uuid.uuid4())

    import re

    match = re.search(r"ORD-\d{6}", user_message)

    if not match:
        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "status": "NEED_ORDER_ID",
            "answer": "Please provide a valid order ID.",
            "llm_calls": 0,
            "tokens_used": 0
        }

    # PHASE 1: Initial planning.
    planning = create_plan(user_message)

    plan = planning["plan"]
    llm_calls = planning["llm_calls"]
    tokens_used = planning["tokens_used"]

    # PHASE 2: Execute the plan.
    execution = execute_plan(plan)

    # PHASE 3: Replan if an unexpected observation occurs.
    if execution["status"] == "UNEXPECTED":

        order_result = execution["results"].get("get_order", {})

        if not order_result or "error" in order_result:
            observation_status = "NOT_FOUND"
        else:
            observation_status = "DENIED"

        observation = {
            "status": observation_status,
            "details": execution.get("error")
        }

        print("Unexpected observation:", observation)
        print("Replanning...")

        replanning = replan(json.dumps(observation))

        revised_plan = replanning["plan"]
        llm_calls += replanning["llm_calls"]
        tokens_used += replanning["tokens_used"]

        for step in revised_plan:
            if step.get("action") == "finish":
                return {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "status": step["status"],
                    "answer": step["reason"],
                    "plan": plan,
                    "revised_plan": revised_plan,
                    "results": execution["results"],
                    "llm_calls": llm_calls,
                    "tokens_used": tokens_used
                }

        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "status": "REPLAN_FAILED",
            "answer": "The revised plan could not be completed.",
            "plan": plan,
            "revised_plan": revised_plan,
            "results": execution["results"],
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    if execution["status"] != "COMPLETED":
        return {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "status": execution["status"],
            "answer": execution.get("error", "Execution failed."),
            "plan": plan,
            "results": execution.get("results", {}),
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    # PHASE 4: Execute the final financial decision.
    decision = decide_next_action(execution["results"])

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": decision["action"],
        "answer": decision["reason"],
        "plan": plan,
        "results": execution["results"],
        "decision": decision,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }

if __name__ == "__main__":

    test_cases = [
    "My order ORD-100048 reached me very late. I would like a refund.",
    "My order ORD-100067 reached me very late. I would like a refund.",
    "Order ORD-100145 arrived far too late, please refund me.",
    "Please refund order ORD-100001, I did not like the delivery.",
    "I want my money back for order ORD-100025.",
    "Refund for order ORD-999999 please.",
    "[LOOP] Refund order ORD-100048."
]

    for message in test_cases:
        print("\n" + "=" * 60)
        print("CUSTOMER:", message)

        result = run_plan_execute(message)

        print("STATUS:", result["status"])
        print("ANSWER:", result["answer"])
        print("LLM CALLS:", result["llm_calls"])
        print("TOKENS USED:", result["tokens_used"])    
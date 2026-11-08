
import json
import time
import uuid
from pathlib import Path

import mock_llm
from policy_tools import TOOL_SPECS, TOOL_FUNCTIONS


# Project paths
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
STATE_DIR = PROJECT_DIR / "state"

AGENT_LOG_PATH = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """
You are an AGENT for customer support at Instamart.

Process customer refund requests using the available tools.

Rules:
1. Retrieve the order details before making a refund decision.
2. Search the relevant policy documents before deciding.
3. Only completed orders are eligible for refunds.
4. A late-delivery refund is available when delivery exceeds 45 minutes.
5. Refunds up to INR 1000 can be issued automatically.
6. Refunds above INR 1000 must be escalated to a human support lead.
7. Never invent order details or policy information.
8. If an order cannot be found, return NOT_FOUND.
9. If a refund is not eligible, return DENIED.
10. If you cannot determine the answer, provide a safe fallback.

Use the available tools to complete the task.
"""


def _append_jsonl(path, entry):
    """Append one JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

def _save_checkpoint(
    run_id,
    scenario_id,
    messages,
    total_tokens,
    step_number,
    steps
):
    """
    Save the agent's current state to a JSON checkpoint file.
    """

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "total_tokens": total_tokens,
        "step_number": step_number,
        "steps": steps,
    }

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2)

    return checkpoint_path

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_step=None
):
    """
    Run a ReAct-style agent.

    The agent alternates between:
    - Reasoning through the LLM
    - Executing tools
    - Observing tool results
    """

    run_id = str(uuid.uuid4())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    steps = []
    total_tokens = 0
    start_time = time.monotonic()

    final_answer = None
    status = "running"

    for iteration in range(max_iterations):

        # Check the time limit before making an LLM call.
        if time.monotonic() - start_time >= time_limit_seconds:
            final_answer = "FALLBACK_TIMEOUT"
            status = "timeout"
            break

        # Call the mock LLM.
        try:
            response = mock_llm.chat(
                messages,
                tools=TOOL_SPECS
            )
        except mock_llm.TransientLLMError:
            final_answer = "FALLBACK_LLM_ERROR"
            status = "llm_error"
            break

        # Track token usage.
        call_tokens = response.usage.total_tokens

        if total_tokens + call_tokens > token_limit:
            final_answer = "FALLBACK_TOKEN_LIMIT"
            status = "token_limit"
            break

        total_tokens += call_tokens

        assistant_message = response.choices[0].message

        # If the LLM returns a final answer, stop.
        
        if response.choices[0].finish_reason != "tool_calls":
            final_answer = assistant_message.content or ""
            status = "completed"

            messages.append({
                "role": "assistant",
                "content": final_answer
            })

        # Preserve the assistant tool-call message.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                    }
                }
                for call in assistant_message.tool_calls
            ]
        })

        _save_checkpoint(
            run_id=run_id,
            scenario_id=scenario_id,
            messages=messages,
            total_tokens=total_tokens,
            step_number=len(steps),
            steps=steps
        )
        if crash_after_step is not None and len(steps) == crash_after_step:
            print(f"Simulated crash after step {len(steps)}")
            return {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "answer": "SIMULATED_CRASH",
                "status": "crashed",
                "steps": steps,
                "total_tokens": total_tokens
            }

        # Execute each requested tool.
        for call in assistant_message.tool_calls:

            tool_name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_FUNCTIONS:
                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }
            else:
                try:
                    observation = TOOL_FUNCTIONS[tool_name](**arguments)
                except Exception as error:
                    observation = {
                        "error": str(error)
                    }

            # Record the tool execution and its observation.
            step = {
                "step": len(steps) + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)

            # Append the observation to the conversation.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            # Log the observation.
            _append_jsonl(
                AGENT_LOG_PATH,
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    **step
                }
            )

    # If the loop finishes without a final answer, use a fallback.
    if final_answer is None:
        final_answer = "FALLBACK_MAX_ITERATIONS"
        status = "max_iterations"

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "answer": final_answer,
        "status": status,
        "steps": steps,
        "total_tokens": total_tokens
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")

    with checkpoint_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    steps = checkpoint["steps"]
    total_tokens = checkpoint["total_tokens"]

    print(f"Resuming run: {run_id}")
    print(f"Previously completed steps: {len(steps)}")

    start_time = time.monotonic()
    final_answer = None
    status = "running"

    for iteration in range(max_iterations):

        if time.monotonic() - start_time >= time_limit_seconds:
            final_answer = "FALLBACK_TIMEOUT"
            status = "timeout"
            break

        try:
            response = mock_llm.chat(messages, tools=TOOL_SPECS)
        except mock_llm.TransientLLMError:
            final_answer = "FALLBACK_LLM_ERROR"
            status = "llm_error"
            break

        call_tokens = response.usage.total_tokens

        if total_tokens + call_tokens > token_limit:
            final_answer = "FALLBACK_TOKEN_LIMIT"
            status = "token_limit"
            break

        total_tokens += call_tokens
        assistant_message = response.choices[0].message

        if response.choices[0].finish_reason != "tool_calls":
            final_answer = assistant_message.content or ""
            status = "completed"

            messages.append({
                "role": "assistant",
                "content": final_answer
            })
            break

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                    }
                }
                for call in assistant_message.tool_calls
            ]
        })

        for call in assistant_message.tool_calls:

            tool_name = call.name
            arguments = call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            try:
                observation = TOOL_FUNCTIONS[tool_name](**arguments)
            except Exception as error:
                observation = {"error": str(error)}

            step = {
                "step": len(steps) + 1,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            _append_jsonl(
                AGENT_LOG_PATH,
                {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    **step
                }
            )

            _save_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                total_tokens=total_tokens,
                step_number=len(steps),
                steps=steps
            )

    if final_answer is None:
        final_answer = "FALLBACK_MAX_ITERATIONS"
        status = "max_iterations"

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "answer": final_answer,
        "status": status,
        "steps": steps,
        "total_tokens": total_tokens
    }

if __name__ == "__main__":
    mock_llm.reset()

    mock_llm._state["fail_next"] = True

    result = run_agent(
        user_message="Please refund my order ORD-110093.",
        scenario_id="TEST-LLM-ERROR"
    )

    print(json.dumps(result, indent=2))
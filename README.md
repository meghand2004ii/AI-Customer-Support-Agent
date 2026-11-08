# AI-Powered Customer Support Agent

## Overview

This project implements an AI-powered customer support agent for an Instamart-style platform. It handles customer refund requests using order information, company policies, and tool-based execution.

Two agent architectures are implemented and evaluated:

- **ReAct Agent:** Reasons and executes tools iteratively.
- **Plan-and-Execute Agent:** Generates an execution plan and performs the required actions sequentially.

The system uses Python, SQLite, TF-IDF-based policy retrieval, and a mock LLM. No real LLM API calls are required.

## Project Structure

```text
Capstone_Project/
├── data/
│   ├── raw/
│   │   └── agent_scenarios.csv
│   ├── policies/
│   └── instamart.db
├── outputs/
│   ├── agent_evaluation.json
│   ├── plan_execute_evaluation.json
│   ├── agent_runs.jsonl
│   ├── plan_execute_runs.jsonl
│   └── refund_ledger.jsonl
├── state/
├── policy_rag.py
├── policy_tools.py
├── mock_llm.py
├── agent.py
├── plan_execute_agent.py
├── evaluate_agent.py
├── evaluate_plan_execute.py
├── requirements.txt
└── README.md
```

## Reproduce

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the ReAct evaluation:

```bash
python evaluate_agent.py
```

5. Run the Plan-and-Execute evaluation:

```bash
python evaluate_plan_execute.py
```

The evaluation reports are saved in the `outputs/` directory.

## Part A — Data and Database

The project uses a SQLite database to store customer, product, and order information. The database supports order lookup and provides the information required for refund eligibility checks.

## Part B — Tools and Policy Retrieval

The system implements tools for:

- Retrieving order details.
- Searching company policies.
- Calculating refund eligibility.
- Issuing refunds.
- Escalating large refunds for human approval.

Policy documents are divided into overlapping chunks and indexed using TF-IDF retrieval. Retrieved policy information is used to support customer responses.

## Part C — ReAct Agent

The ReAct agent follows an iterative reasoning and tool-execution approach. It observes tool results before deciding its next action.

Implemented capabilities include:

- Tool selection and execution.
- Refund and escalation decisions.
- Maximum iteration limits.
- Token and execution-time limits.
- Error handling and safe fallbacks.
- Checkpoint creation and execution resumption.
- Idempotency protection for financial actions.

## Part D — Plan-and-Execute Agent and Evaluation

The Plan-and-Execute agent creates an ordered plan before executing the required tools. It can revise its plan when unexpected observations occur.

### Evaluation Results

Seven scenarios were used to evaluate both architectures.

| Scenario | ReAct | Plan-and-Execute |
|---|---|---|
| S1 | REFUND | REFUND |
| S2 | REFUND | REFUND |
| S3 | ESCALATE | ESCALATE |
| S4 | DENIED | DENIED |
| S5 | DENIED | DENIED |
| S6 | NOT_FOUND | NOT_FOUND |
| S7 | FALLBACK | REFUND |

### Performance Summary

| Metric | ReAct | Plan-and-Execute |
|---|---:|---:|
| Scenarios passed | 7/7 | 6/7 |
| Accuracy | 100% | 85.71% |
| Total tokens | 16,139 | 633 |

### Architecture Comparison

ReAct provides adaptive execution by selecting tools based on intermediate observations. It successfully detected repeated tool execution in S7 and stopped using its maximum-iteration fallback.

Plan-and-Execute uses a predefined execution plan and supports replanning when unexpected observations occur. It consumed substantially fewer tokens in the mock evaluation. However, it did not detect the repeated-tool scenario and processed the refund instead.

ReAct is preferred for this refund workflow when safety and adaptability are priorities, while Plan-and-Execute demonstrates better token efficiency in this implementation.

### Safety and Recovery Testing

The ReAct agent was tested for:

- Maximum iteration exhaustion.
- Token budget exhaustion.
- Execution timeout.
- Simulated LLM failures.
- Crash recovery using saved checkpoints.
- Duplicate financial action prevention.

## Limitations and Next Steps

- The LLM is simulated using a mock implementation; no real LLM API is connected.
- Policy retrieval uses TF-IDF rather than semantic embeddings.
- The database and refund ledger are local development implementations.
- Production deployment would require authentication, authorization, persistent transaction handling, monitoring, and stronger concurrency-safe idempotency.
- Future improvements include semantic policy retrieval, integration with a real LLM, and more comprehensive evaluation scenarios.

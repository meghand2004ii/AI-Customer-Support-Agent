import sqlite3
from pathlib import Path
from policy_rag import PolicyRetriever, load_policy_chunks

import sys
sys.path.insert(0, str(Path(__file__).parent))
from policy_rag import PolicyRetriever, load_policy_chunks
import sqlite3
from pathlib import Path

import json
from datetime import datetime, timezone


DB_PATH = Path(__file__).parent / "data" / "instamart.db"

TOOL_SPECS = [
    {
        "name": "get_order",
        "description": "Retrieve order details using an order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up."
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The policy question or search query."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a late-delivery refund based on the order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "The total order amount in INR."
                },
                "delivery_minutes": {
                    "type": ["number", "null"],
                    "description": "The delivery duration in minutes."
                }
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an approved refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID receiving the refund."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The approved refund amount in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund that requires human support approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID requiring human review."
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The proposed refund amount in INR."
                }
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

def get_order(order_id):
    """
    Retrieve order details from SQLite.

    Returns:
        dict: Order details if found, otherwise an error message.
    """
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": f"Order {order_id} not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }

    finally:
        connection.close()


# Load policy documents and initialize the retriever
policy_chunks = load_policy_chunks()
policy_retriever = PolicyRetriever(policy_chunks)


def search_policy(query):
    """
    Search policy documents and return the top 3 relevant chunks.
    """
    results = policy_retriever.retrieve(query, k=3)

    return {
        "query": query,
        "results": results
    }


def calculate_refund(order_amount, delivery_minutes):
    """
    Calculate the refund amount based on the late-delivery policy.
    """
    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery time is unavailable."
        }

    if delivery_minutes > 45:
        return {
            "refund_inr": round(float(order_amount), 2),
            "reason": "Full refund for delivery exceeding 45 minutes."
        }

    return {
        "refund_inr": 0.0,
        "reason": "No late-delivery refund for delivery within 45 minutes."
    }


LEDGER_PATH = Path(__file__).parent / "outputs" / "refund_ledger.jsonl"



def _record_refund_action(tool_name, order_id, refund_inr):
    """
    Record a financial action exactly once per order and tool.
    Prevent duplicate refund or escalation records.
    """
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    amount = round(float(refund_inr), 2)

    # Check whether this action was already recorded.
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                existing = json.loads(line)

                if (
                    existing["tool"] == tool_name
                    and existing["order_id"] == order_id
                ):
                    return {
                        "order_id": order_id,
                        "refund_inr": existing["refund_inr"],
                        "status": "already_recorded",
                        "tool": tool_name,
                    }

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": amount,
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "order_id": order_id,
        "refund_inr": amount,
        "status": "recorded",
        "tool": tool_name,
    }



def issue_refund(order_id, refund_inr):
    """
    Record an approved refund.
    """
    return _record_refund_action(
        "issue_refund",
        order_id,
        refund_inr
    )


def escalate_to_human(order_id, refund_inr):
    """
    Record a refund that requires human approval.
    """
    return _record_refund_action(
        "escalate_to_human",
        order_id,
        refund_inr
    )


if __name__ == "__main__":
    print("--- Testing get_order ---")
    print(get_order("ORD-110093"))

    print("\n--- Testing search_policy ---")
    result = search_policy("refund policy late delivery")

    for item in result["results"]:
        print(
            f"\nSource: {item['source']}"
            f"\nScore: {item['score']:.4f}"
            f"\nText: {item['text']}"
        )

    print("\n--- Testing calculate_refund ---")

    print("Late delivery:")
    print(calculate_refund(127.50, 50))

    print("\nOn-time delivery:")
    print(calculate_refund(127.50, 30))

    print("\nMissing delivery time:")
    print(calculate_refund(127.50, None))

    print("\n--- Testing issue_refund ---")
    print(issue_refund("ORD-TEST-001", 128.25))

    print("\n--- Testing escalate_to_human ---")
    print(escalate_to_human("ORD-TEST-002", 1999.00))

TOOL_FUNCTIONS = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}
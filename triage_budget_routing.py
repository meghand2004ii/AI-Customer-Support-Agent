
import pandas as pd

from triage import (
    process_tickets,
    run_classification,
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
)

# Part C - Budget Demo

TOKEN_BUDGET = 6000


def run_budget_demo(tickets):
    """
    Run zero-shot classification with a cumulative token budget.
    If usage has already reached the budget before a model call,
    the ticket is routed to human review.
    """
    import mock_llm

    mock_llm.reset()

    eligible = tickets[tickets["status"] == "eligible"]

    results = []
    running_tokens = 0

    for _, row in eligible.iterrows():

        # Check budget before making the next model call.
        if running_tokens >= TOKEN_BUDGET:
            results.append({
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "status": "budget_exhausted",
                "category": None,
                "tokens_used": 0,
                "route": "human_review",
            })
            continue

        result = None

        # Use the existing retry and output-validation logic.
        from triage import classify_ticket

        result = classify_ticket(
            ticket_id=row["ticket_id"],
            redacted_text=row["redacted_text"],
            system_prompt=ZERO_SHOT_PROMPT,
        )

        running_tokens += result["tokens_used"]

        if result["status"] in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        ]:
            route = "human_review"
        else:
            route = "support_queue"

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": result["status"],
            "category": result["category"],
            "tokens_used": result["tokens_used"],
            "route": route,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "outputs/triage_budget_demo.csv",
        index=False,
    )

    print("\n===== BUDGET DEMO =====")
    print("Token budget:", TOKEN_BUDGET)
    print("Tokens consumed:", running_tokens)
    print("Budget-exhausted tickets:",
          (results_df["status"] == "budget_exhausted").sum())
    print("Status counts:")
    print(results_df["status"].value_counts().to_string())

    return results_df


# Part C - Final Few-shot Routing

FINAL_TOKEN_BUDGET = 100000


def run_final_triage(tickets):
    """
    Run few-shot classification and route tickets according
    to status, category, and customer segment.
    """
    import mock_llm

    mock_llm.reset()

    eligible = tickets[tickets["status"] == "eligible"]

    results = []

    running_tokens = 0

    for _, row in eligible.iterrows():

        # The final budget is 100,000 tokens.
        if running_tokens >= FINAL_TOKEN_BUDGET:
            results.append({
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "status": "budget_exhausted",
                "category": None,
                "tokens_used": 0,
            })
            continue

        from triage import classify_ticket

        result = classify_ticket(
            ticket_id=row["ticket_id"],
            redacted_text=row["redacted_text"],
            system_prompt=FEW_SHOT_PROMPT,
        )

        running_tokens += result["tokens_used"]

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": result["status"],
            "category": result["category"],
            "tokens_used": result["tokens_used"],
        })

    results_df = pd.DataFrame(results)

    # Include all tickets, including skipped_empty and flagged_long.
    non_eligible = tickets[tickets["status"] != "eligible"]

    non_eligible_results = pd.DataFrame({
        "ticket_id": non_eligible["ticket_id"],
        "customer_id": non_eligible["customer_id"],
        "status": non_eligible["status"],
        "category": None,
        "tokens_used": 0,
    })

    results_df = pd.concat(
        [results_df, non_eligible_results],
        ignore_index=True,
    )

    # Load Part B customer segments.
    segments = pd.read_csv("outputs/customer_segments.csv")

    segments = segments[
        ["customer_id", "segment_name", "is_premium"]
    ]

    # Join customer segment information.
    results_df = results_df.merge(
        segments,
        on="customer_id",
        how="left",
    )

    # Missing customers are treated as non-premium.
    results_df["is_premium"] = (
        results_df["is_premium"]
        .fillna(False)
        .astype(bool)
    )

    results_df["segment_name"] = (
        results_df["segment_name"]
        .fillna("")
    )

    # Apply routing and priority rules.
    def assign_route(row):
        if row["status"] in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        ]:
            return pd.Series({
                "route": "human_review",
                "priority": "NORMAL",
            })

        if row["category"] == "fraud":
            return pd.Series({
                "route": "security_team",
                "priority": "HIGH",
            })

        if row["is_premium"]:
            return pd.Series({
                "route": "premium_desk",
                "priority": "HIGH",
            })

        return pd.Series({
            "route": "support_queue",
            "priority": "NORMAL",
        })

    results_df[["route", "priority"]] = results_df.apply(
        assign_route,
        axis=1,
    )

    # Final required columns.
    results_df = results_df[
        [
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    ]

    results_df.to_csv(
        "outputs/triage_results.csv",
        index=False,
    )

    print("\n===== FINAL TRIAGE =====")
    print("Tickets:", len(results_df))
    print("Tokens consumed:", running_tokens)
    print("Status counts:")
    print(results_df["status"].value_counts().to_string())

    print("\nRouting counts:")
    print(results_df["route"].value_counts().to_string())

    print("\nPriority counts:")
    print(results_df["priority"].value_counts().to_string())

    return results_df


# Main

if __name__ == "__main__":

    tickets = process_tickets()

    budget_results = run_budget_demo(tickets)

    final_results = run_final_triage(tickets)

    print("\nSaved:")
    print("outputs/triage_budget_demo.csv")
    print("outputs/triage_results.csv")

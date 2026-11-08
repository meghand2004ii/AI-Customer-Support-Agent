import re
import pandas as pd
import mock_llm

INPUT_FILE = "data/raw/tickets.csv"

EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_RE = r"\b[6-9]\d{9}\b"
ORDER_RE = r"ORD-\d{6}"

def redact_text(text):
    """Redact email, phone, and order ID in the required order."""
    if pd.isna(text):
        text = ""

    text = str(text)

    text = re.sub(EMAIL_RE, "[EMAIL]", text)
    text = re.sub(PHONE_RE, "[PHONE]", text)
    text = re.sub(ORDER_RE, "[ORDER_ID]", text)

    return text


CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_PROMPT = """
You are a support ticket classification assistant.

CLASSIFY the customer ticket into exactly one of these categories:
billing, delivery, technical, fraud.

Return only valid JSON in this exact format:
{"category":"<category>","reason":"<brief reason>"}

The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON.
""".strip()


FEW_SHOT_PROMPT = """
You are a support ticket classification assistant.

CLASSIFY the customer ticket into exactly one of these categories:
billing, delivery, technical, fraud.

Return only valid JSON in this exact format:
{"category":"<category>","reason":"<brief reason>"}

The category must be one of the four allowed categories.
Do not include markdown or any text outside the JSON.

EXAMPLES:

Ticket: I was charged twice for the same order.
{"category":"billing","reason":"The customer reports a duplicate charge."}

Ticket: My order has not arrived and is delayed.
{"category":"delivery","reason":"The customer reports a delivery delay."}

Ticket: The app crashes when I try to make a payment.
{"category":"technical","reason":"The customer reports an application error."}

Ticket: Someone used my card without my permission.
{"category":"fraud","reason":"The customer reports an unauthorized transaction."}
""".strip()


import json
import random
import time

BASE_DELAY = 0.05
MAX_RETRIES = 3


def classify_ticket(ticket_id, redacted_text, system_prompt):
    """
    Classify one redacted ticket.
    The user message sent to the mock model is exactly the redacted text.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            response_text = response.choices[0].message.content
            usage_tokens = response.usage.total_tokens

            # First: parse JSON.
            try:
                parsed = json.loads(response_text)

                if (
                    isinstance(parsed, dict)
                    and parsed.get("category") in CATEGORIES
                    and isinstance(parsed.get("reason"), str)
                ):
                    return {
                        "status": "classified_json",
                        "category": parsed["category"],
                        "reason": parsed["reason"],
                        "tokens_used": usage_tokens,
                    }

                # Valid JSON but an invalid category must not be guessed.
                return {
                    "status": "flagged_invalid_output",
                    "category": None,
                    "reason": None,
                    "tokens_used": usage_tokens,
                }

            except json.JSONDecodeError:
                # Second: keyword fallback.
                words = re.findall(r"[a-z]+", response_text.lower())
                found_categories = [
                    category for category in CATEGORIES
                    if category in words
                ]

                if len(found_categories) == 1:
                    return {
                        "status": "classified_keyword",
                        "category": found_categories[0],
                        "reason": response_text,
                        "tokens_used": usage_tokens,
                    }

                return {
                    "status": "flagged_invalid_output",
                    "category": None,
                    "reason": None,
                    "tokens_used": usage_tokens,
                }

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                return {
                    "status": "flagged_invalid_output",
                    "category": None,
                    "reason": None,
                    "tokens_used": 0,
                }

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: ticket={ticket_id}, "
                f"retry_number={retry_number}, "
                f"wait={wait_seconds:.4f}s"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected classifier exit")


def run_classification(df, system_prompt, run_name):
    """Run classification on all eligible tickets and collect metrics."""
    mock_llm.reset()

    results = []

    eligible = df[df["status"] == "eligible"]

    for _, row in eligible.iterrows():
        result = classify_ticket(
            ticket_id=row["ticket_id"],
            redacted_text=row["redacted_text"],
            system_prompt=system_prompt,
        )

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "status": result["status"],
            "category": result["category"],
            "reason": result["reason"],
            "tokens_used": result["tokens_used"],
        })

    results_df = pd.DataFrame(results)

    # Accuracy is calculated only for successfully classified tickets.
    classified = results_df[
        results_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if len(classified) > 0
        else 0
    )

    total_tokens = results_df["tokens_used"].sum()

    print(f"\n===== {run_name} RESULTS =====")
    print("Eligible tickets:", len(eligible))
    print("Classified tickets:", len(classified))
    print("Accuracy:", f"{accuracy:.2%}")
    print("Total usage tokens:", total_tokens)
    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) > 0
        else 0,
    )

    print("\nStatus counts:")
    print(results_df["status"].value_counts().to_string())

    print("\nConfusion matrix:")
    print(
        pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )

    return results_df


def run_part_c_classification():
    tickets = process_tickets()

    # Independent zero-shot run.
    zero_shot_results = run_classification(
        tickets,
        ZERO_SHOT_PROMPT,
        "ZERO-SHOT",
    )

    # Independent few-shot run.
    few_shot_results = run_classification(
        tickets,
        FEW_SHOT_PROMPT,
        "FEW-SHOT",
    )

    # Save the comparison results.
    zero_shot_results.to_csv(
        "outputs/triage_zero_shot.csv",
        index=False,
    )

    few_shot_results.to_csv(
        "outputs/triage_few_shot.csv",
        index=False,
    )

    print("\nSaved:")
    print("outputs/triage_zero_shot.csv")
    print("outputs/triage_few_shot.csv")


def process_tickets():
    df = pd.read_csv(INPUT_FILE)

    # Redact all ticket text before any model call.
    df["redacted_text"] = df["text"].apply(redact_text)

    # Estimate token count using the supplied mock LLM function.
    df["estimated_tokens"] = df["redacted_text"].apply(
        mock_llm.est_tokens
    )

    # Required guardrails.
    df["status"] = "eligible"

    empty_mask = df["redacted_text"].str.strip() == ""
    long_mask = df["estimated_tokens"] > 400

    df.loc[empty_mask, "status"] = "skipped_empty"
    df.loc[long_mask & ~empty_mask, "status"] = "flagged_long"

    # Print useful validation results.
    print("Total tickets:", len(df))
    print("Empty tickets:", empty_mask.sum())
    print("Long tickets:", (long_mask & ~empty_mask).sum())
    print("Eligible tickets:", (df["status"] == "eligible").sum())

    print("\nEstimated token count for long tickets:")
    print(df.loc[long_mask, ["ticket_id", "estimated_tokens"]].to_string(index=False))

    print("\nRedaction counts:")
    print("Emails:", df["text"].fillna("").str.count(EMAIL_RE).sum())
    print("Phones:", df["text"].fillna("").str.count(PHONE_RE).sum())
    print("Order IDs:", df["text"].fillna("").str.count(ORDER_RE).sum())

    # Verify that no original sensitive patterns remain in redacted text.
    remaining_emails = df["redacted_text"].str.count(EMAIL_RE).sum()
    remaining_phones = df["redacted_text"].str.count(PHONE_RE).sum()
    remaining_orders = df["redacted_text"].str.count(ORDER_RE).sum()

    print("\nRemaining sensitive patterns:")
    print("Emails:", remaining_emails)
    print("Phones:", remaining_phones)
    print("Order IDs:", remaining_orders)

    assert remaining_emails == 0
    assert remaining_phones == 0
    assert remaining_orders == 0

    return df


if __name__ == "__main__":
    run_part_c_classification()
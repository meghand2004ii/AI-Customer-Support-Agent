
from pathlib import Path

import mock_llm
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


POLICY_DIR = Path("data/policies")
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def chunk_text(text, chunk_size=40, overlap=0):
    """
    Split text into chunks of chunk_size words with the specified overlap.
    """
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def load_policy_chunks():
    """
    Load all policy text files and create source-tagged chunks.
    """
    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append({
                "source": file_path.name,
                "text": chunk,
            })

    return all_chunks


class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        self.chunk_texts = [
            chunk["text"] for chunk in chunks
        ]

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.chunk_texts
        )

    def retrieve(self, query, k=3):
        """
        Return the top-k policy chunks ranked by cosine similarity.
        """
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            })

        return results


def answer(question, retriever):
    """
    Answer a policy question using the top 3 retrieved chunks.
    Return the answer text and the source filenames.
    """
    retrieved = retriever.retrieve(question, k=3)

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a POLICY ASSISTANT.

Answer the user's question using only the supplied policy context.
Do not use outside knowledge.
If the context does not contain the answer, respond with:
I could not find the answer in the provided policy documents.
""".strip()

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    response = mock_llm.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    return {
        "answer": answer_text,
        "sources": sources,
    }


def evaluate_policy_rag(
    retriever,
    eval_path="data/raw/policy_eval.csv"
):
    """
    Evaluate retrieval, answer fact matching, and fallback behavior.
    """
    df = pd.read_csv(eval_path)

    retrieval_hits = 0
    fact_matches = 0
    fallback_matches = 0

    answerable_count = 0
    unanswerable_count = 0

    results = []

    for _, row in df.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        retrieved = retriever.retrieve(question, k=3)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        result = answer(question, retriever)

        response = result["answer"]
        sources = result["sources"]

        # Initialize metrics for every question
        retrieval_correct = False
        fact_correct = False
        fallback_correct = None

        if expected_doc != "NONE":
            answerable_count += 1

            retrieval_correct = (
                expected_doc in retrieved_sources
            )

            if retrieval_correct:
                retrieval_hits += 1

            fact_correct = (
                isinstance(expected_fact, str)
                and expected_fact.lower() in response.lower()
            )

            if fact_correct:
                fact_matches += 1

        else:
            unanswerable_count += 1

            fallback_correct = (
                response.strip().lower().startswith(
                    "i could not find"
                )
            )

            if fallback_correct:
                fallback_matches += 1

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_sources": ", ".join(
                retrieved_sources
            ),
            "retrieval_correct": retrieval_correct,
            "expected_fact": expected_fact,
            "response": response,
            "fact_correct": fact_correct,
            "fallback_correct": fallback_correct,
        })

    results_df = pd.DataFrame(results)

    Path("outputs").mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        "outputs/policy_eval_results.csv",
        index=False
    )

    print("\n--- Policy RAG Evaluation ---")
    print(f"Answerable questions: {answerable_count}")
    print(
        f"Retrieval hits: {retrieval_hits}/{answerable_count}"
    )
    print(
        f"Fact matches: {fact_matches}/{answerable_count}"
    )
    print(
        f"Unanswerable questions: {unanswerable_count}"
    )
    print(
        f"Correct fallbacks: {fallback_matches}/{unanswerable_count}"
    )

    return results_df


if __name__ == "__main__":
    chunks = load_policy_chunks()

    retriever = PolicyRetriever(chunks)

    evaluate_policy_rag(retriever)

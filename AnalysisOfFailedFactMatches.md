### Analysis of Failed Fact Matches

In the zero-overlap experiment, the Policy RAG system achieved 12/12 retrieval hits but only 9/12 fact matches. The three failed questions concerned the full-refund eligibility time, approval of refunds above INR 1000, and the return time for failed payments.

The retrieved sources included the expected policy documents, indicating that the retrieval stage was successful. However, the generated responses either omitted the expected facts or included incomplete or unrelated information.

This demonstrates that retrieving the correct document does not necessarily guarantee a correct answer. Chunk boundaries can separate important conditions from the corresponding policy statements. A 10-word overlap helps preserve contextual continuity between chunks and improves the likelihood that the retrieved context contains the complete information required to answer a question.
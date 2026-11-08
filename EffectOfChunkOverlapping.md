### Effect of Chunk Overlap on Policy RAG

An experiment was conducted to evaluate the effect of chunk overlap on the Policy RAG system. The chunk size was maintained at 40 words, while the overlap was changed from 10 words to 0 words.

With an overlap of 10 words, the system achieved 12/12 retrieval hits and 12/12 fact matches. With no overlap, retrieval remained at 12/12, but fact matching decreased to 9/12. Both configurations correctly handled the two unanswerable questions, achieving 2/2 correct fallbacks.

The results indicate that chunk overlap improves answer quality by preserving contextual information across chunk boundaries. Although the correct policy documents were retrieved in both configurations, some facts were not fully available in the retrieved context when overlap was removed. Therefore, a 10-word overlap was more effective for this policy dataset.
import google.generativeai as genai
from backend.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

class Reranker:
    def __init__(self, embedder):
        self.embedder = embedder

    def rerank(self, query, documents, top_k=4):
        try:
            query_emb = self.embedder.embed_query(query)
            doc_embs = [self.embedder.embed_query(doc.page_content) for doc in documents]

            def cosine(a, b):
                return sum(x * y for x, y in zip(a, b))

            scored = [
                (doc, cosine(query_emb, emb))
                for doc, emb in zip(documents, doc_embs)
            ]

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        except Exception as e:
            print("Reranker error:", e)
            return [(doc, 0) for doc in documents[:top_k]]

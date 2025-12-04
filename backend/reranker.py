from google import genai
from backend.config import GOOGLE_API_KEY
from backend.reranker import Reranker
client = genai.Client(api_key=GOOGLE_API_KEY)
class KnowledgeAssistant:
    def __init__(self, vectorstore, model_name="gemini-2.5-flash"):
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.embedder = vectorstore.embeddings
        self.reranker = Reranker(self.embedder)
    def rewrite_query(self, query: str) -> str:
        try:
            prompt = f"""
Rewrite the following question into a short, search-optimized query.
Keep it concise and factual.
Original: {query}
Rewritten:
"""
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
            )
            rewritten = response.text.strip()
            if len(rewritten) < 3:
                return query
            return rewritten
        except:
            return query.strip()
    def query(self, question: str, k: int = 4):
        try:
            rewritten = self.rewrite_query(question)
            initial_k = max(k * 5, 10)
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": initial_k}
            )
            retrieved_docs = retriever.invoke(rewritten)
            if not retrieved_docs:
                retrieved_docs = retriever.invoke(question)
            reranked = self.reranker.rerank(rewritten, retrieved_docs, top_k=k)
            top_docs = [d for d, score in reranked]
            context_parts = []
            for i, doc in enumerate(top_docs):
                filename = doc.metadata.get("source", doc.metadata.get("filename", "Unknown"))
                chunk_header = f"[SOURCE: {filename}, chunk {i+1}]"
                context_parts.append(f"{chunk_header}\n{doc.page_content}\n")
            context = "\n\n".join(context_parts)
            prompt = f"""
You are an accurate assistant.

Use ONLY the provided context to answer the user's question.
If the answer is not contained in the context, say "I don't know".

Always cite sources exactly as [SOURCE: filename].

Context:
{context}

Question:
{question}

Answer:
"""
            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(temperature=0.1)
            )
            answer_text = response.text
            return {
                "answer": answer_text,
                "sources": [
                    {
                        "content": d.page_content,
                        "source": d.metadata.get("source", "Unknown"),
                        "filename": d.metadata.get("filename", "Unknown"),
                    }
                    for d in top_docs
                ],
            }
        except Exception as e:
            print(f" Error in query(): {e}")
            raise e

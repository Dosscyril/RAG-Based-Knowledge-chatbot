import google.generativeai as genai
from backend.config import GOOGLE_API_KEY
from backend.reranker import Reranker

# Configure API
genai.configure(api_key=GOOGLE_API_KEY)

class KnowledgeAssistant:
    def __init__(self, vectorstore, model_name="gemini-1.5-flash"):
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.embedder = vectorstore.embeddings
        self.reranker = Reranker(self.embedder)

    def rewrite_query(self, query: str) -> str:
        prompt = f"""
Rewrite the following question into a short, search-optimized phrase.
Keep it concise and factual.

Original: {query}
Rewritten:
"""
        try:
            response = self.model.generate_content(prompt)
            rewritten = response.text.strip()
            return rewritten if len(rewritten) > 2 else query
        except Exception as e:
            print("rewrite_query error:", e)
            return query

    def query(self, question: str, k: int = 4):
        try:
            rewritten = self.rewrite_query(question)

            # Retrieve raw documents
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": max(k * 5, 10)}
            )
            retrieved_docs = retriever.invoke(rewritten)
            if not retrieved_docs:
                retrieved_docs = retriever.invoke(question)

            # Rerank using cosine similarity
            reranked = self.reranker.rerank(
                rewritten, retrieved_docs, top_k=k
            )
            top_docs = [doc for doc, score in reranked]

            # Build context
            context_blocks = []
            for i, doc in enumerate(top_docs):
                src = doc.metadata.get("source", "Unknown")
                context_blocks.append(
                    f"[SOURCE: {src}, chunk {i+1}]\n{doc.page_content}"
                )
            context = "\n\n".join(context_blocks)

            # RAG prompt
            final_prompt = f"""
Use ONLY the provided context to answer the question.
If the answer is not in the context, reply: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

            response = self.model.generate_content(
                final_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )

            return {
                "answer": response.text,
                "sources": [
                    {
                        "content": d.page_content,
                        "filename": d.metadata.get("source", "Unknown"),
                    }
                    for d in top_docs
                ],
            }

        except Exception as e:
            print("query() error:", e)
            return {
                "answer": f"⚠️ Error from Gemini: {e}",
                "sources": []
            }

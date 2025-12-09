import google.generativeai as genai
from backend.config import GOOGLE_API_KEY
from backend.reranker import Reranker

# Configure API
genai.configure(api_key=GOOGLE_API_KEY)

class KnowledgeAssistant:
    def __init__(self, vectorstore, model_name="gemini-2.0-flash-exp"):
        """
        Initialize with Gemini 2.0 Flash (faster & better than 1.5)
        Free tier: 15 RPM, 1M TPM, 1500 RPD
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": 0.2,  # Slightly creative but focused
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        self.embedder = vectorstore.embeddings
        self.reranker = Reranker(self.embedder)

    def rewrite_query(self, query: str) -> str:
        """Optimize query for better retrieval using Gemini 2.0"""
        prompt = f"""You are a search query optimizer. Rewrite the user's question into a concise, keyword-rich search phrase.

Rules:
- Keep it under 10 words
- Focus on key concepts and entities
- Remove filler words
- Make it retrieval-friendly

Original question: {query}

Optimized search query:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=50
                )
            )
            rewritten = response.text.strip()
            # Fallback if response is too short or too long
            if 2 < len(rewritten) < 100:
                return rewritten
            return query
        except Exception as e:
            print(f"Query rewrite error: {e}")
            return query

    def query(self, question: str, k: int = 4):
        """
        Main RAG pipeline with Gemini 2.0 Flash
        
        Args:
            question: User's question
            k: Number of top documents to use for context
        """
        try:
            # Step 1: Optimize query
            rewritten = self.rewrite_query(question)
            print(f"📝 Original: {question}")
            print(f"🔍 Optimized: {rewritten}")

            # Step 2: Retrieve more docs than needed (for reranking)
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": max(k * 3, 12)}  # Get 3x more for reranking
            )
            
            retrieved_docs = retriever.invoke(rewritten)
            
            # Fallback to original query if no results
            if not retrieved_docs:
                print("⚠️ No results for rewritten query, trying original...")
                retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information in the documents.",
                    "sources": []
                }

            # Step 3: Rerank using cosine similarity
            reranked = self.reranker.rerank(rewritten, retrieved_docs, top_k=k)
            top_docs = [doc for doc, score in reranked]
            
            print(f"✅ Retrieved {len(retrieved_docs)} docs, reranked to top {len(top_docs)}")

            # Step 4: Build context with source tracking
            context_blocks = []
            for i, doc in enumerate(top_docs):
                src = doc.metadata.get("source", "Unknown")
                # Clean filename from path
                src_clean = src.split("/")[-1] if "/" in src else src
                context_blocks.append(
                    f"[Document {i+1}: {src_clean}]\n{doc.page_content}\n"
                )
            context = "\n".join(context_blocks)

            # Step 5: RAG prompt optimized for Gemini 2.0
            final_prompt = f"""You are a helpful AI assistant answering questions based on provided documents.

**INSTRUCTIONS:**
1. Answer ONLY using information from the context below
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Be specific and cite which document(s) you're using
4. Keep your answer concise but complete

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWER:**"""

            # Step 6: Generate answer with Gemini 2.0 Flash
            response = self.model.generate_content(final_prompt)

            return {
                "answer": response.text.strip(),
                "sources": [
                    {
                        "content": d.page_content,
                        "filename": d.metadata.get("source", "Unknown").split("/")[-1],
                    }
                    for d in top_docs
                ],
            }

        except Exception as e:
            print(f"❌ Query error: {e}")
            return {
                "answer": f"⚠️ An error occurred: {str(e)}",
                "sources": []
            }
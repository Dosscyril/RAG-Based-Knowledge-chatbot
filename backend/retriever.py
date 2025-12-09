import google.generativeai as genai
from backend.config import GOOGLE_API_KEY
from backend.reranker import Reranker

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class KnowledgeAssistant:
    def __init__(self, vectorstore, model_name: str = "gemini-2.0-flash"):
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.reranker = Reranker(vectorstore.embeddings)

    def expand_query(self, question: str) -> list[str]:
        prompt = f"""
Expand the following question into 5 highly relevant search queries.
These queries help retrieve the right document chunks.

Question: {question}

Return ONLY the queries, one per line. No numbering. No explanations.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.2)
            )
            lines = response.text.strip().split("\n")
            expanded = [l.strip("-• ").strip() for l in lines if l.strip()]
            if question not in expanded:
                expanded.insert(0, question)
            return expanded[:6]
        except Exception as e:
            print(f"Query expansion failed, falling back: {e}")
            return [question]

    def retrieve_chunks(self, expanded_queries, k):
        all_docs = []
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        for q in expanded_queries:
            try:
                docs = retriever.invoke(q)
                all_docs.extend(docs)
            except:
                pass

        # Remove duplicates
        unique_docs = {doc.page_content: doc for doc in all_docs}
        return list(unique_docs.values())

    def query(self, question: str, k: int = 4):
        try:
            expanded_queries = self.expand_query(question)
            relevant_docs = self.retrieve_chunks(expanded_queries, k)

            # If no context found, fallback to general knowledge
            if len(relevant_docs) == 0:
                fallback_prompt = f"""
The user asked:
{question}

There is no document context available.
Answer using general knowledge in a clean structured format:

**📌 Summary**
- Short bullet summary.

**📘 Detailed Breakdown**
- Detailed explanation in bullet points.

**📚 Sources**
- No document sources available.
"""

                response = self.model.generate_content(
                    fallback_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )

                return {"answer": response.text, "sources": []}

            # Build context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"""
You are a highly structured and clean-answering RAG assistant.

Use ONLY the provided context.  
If the answer is not found in context, say:

"I don't know based on the uploaded documents."

Follow this EXACT response format:

----------------
**📌 Summary**
- 1–2 bullet points summarizing the answer.

**📘 Detailed Breakdown**
- Bullet explanation of all key concepts.
- Use **bold** for section headings.
- Include sub-bullets where required.

**🔍 Key Points**
- Short crisp important bullet points.

**📚 Sources**
- filename (chunk_id)
----------------

### CONTEXT:
{context}

### QUESTION:
{question}

Now generate the final structured answer:
"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )

            return {
                "answer": response.text,
                "sources": [
                    {
                        "content": doc.page_content,
                        "filename": doc.metadata.get("source", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "?")
                    }
                    for doc in relevant_docs
                ],
            }

        except Exception as e:
            print(f"Error in query(): {e}")
            return {
                "answer": f"⚠️ Error from Gemini: {e}",
                "sources": []
            }

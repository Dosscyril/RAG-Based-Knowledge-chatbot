from google import genai
from backend.config import GOOGLE_API_KEY
client = genai.Client(api_key=GOOGLE_API_KEY)
class KnowledgeAssistant:
    def __init__(self, vectorstore, model_name: str = "gemini-2.5-flash"):
        self.vectorstore = vectorstore
        self.model_name = model_name
    def expand_query(self, question: str) -> list[str]:
        prompt = f"""
Expand the following question into 5 highly relevant search queries.
These queries help retrieve the right document chunks.

Question: {question}

Return ONLY the queries, one per line. No numbering. No explanations.
"""

        try:
            resp = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(temperature=0.2)
            )
            lines = resp.text.strip().split("\n")
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
        unique_docs = {doc.page_content: doc for doc in all_docs}
        return list(unique_docs.values())
    def query(self, question: str, k: int = 4):
        try:
            expanded_queries = self.expand_query(question)
            relevant_docs = self.retrieve_chunks(expanded_queries, k)
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
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=[fallback_prompt],
                    config=genai.types.GenerateContentConfig(temperature=0.2)
                )
                return {"answer": resp.text, "sources": []}
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

Now generate the final formatted answer:
"""

            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=genai.types.GenerateContentConfig(temperature=0.1)
            )
            return {
                "answer": response.text,
                "sources": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "?")
                    }
                    for doc in relevant_docs
                ],
            }
        except Exception as e:
            print(f" Error in query(): {e}")
            raise e
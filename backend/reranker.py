import numpy as np
from typing import List, Tuple
from langchain_core.documents import Document

class Reranker:
    """
    Semantic reranker using cosine similarity
    Improves retrieval by reordering results based on semantic relevance
    """
    
    def __init__(self, embedder):
        self.embedder = embedder

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors
        Returns: similarity score between -1 and 1 (higher = more similar)
        """
        try:
            # Convert to numpy for faster computation
            a = np.array(vec_a)
            b = np.array(vec_b)
            
            # Compute cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot_product / (norm_a * norm_b))
        
        except Exception as e:
            print(f"⚠️ Cosine similarity error: {e}")
            return 0.0

    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by semantic similarity to query
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        try:
            print(f"🔄 Reranking {len(documents)} documents...")
            
            # Embed query once
            query_emb = self.embedder.embed_query(query)
            
            # Embed all documents (batch processing would be faster but more complex)
            doc_embeddings = []
            for doc in documents:
                try:
                    emb = self.embedder.embed_query(doc.page_content)
                    doc_embeddings.append(emb)
                except Exception as e:
                    print(f"⚠️ Failed to embed document: {e}")
                    doc_embeddings.append([0] * len(query_emb))  # Zero vector as fallback
            
            # Calculate similarity scores
            scored_docs = []
            for doc, doc_emb in zip(documents, doc_embeddings):
                score = self.cosine_similarity(query_emb, doc_emb)
                scored_docs.append((doc, score))
            
            # Sort by score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            top_results = scored_docs[:top_k]
            
            # Print reranking info
            print(f"📊 Top scores: {[f'{score:.3f}' for _, score in top_results[:3]]}")
            
            return top_results

        except Exception as e:
            print(f"❌ Reranker error: {e}")
            # Fallback: return original order with zero scores
            return [(doc, 0.0) for doc in documents[:top_k]]
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.config import COLLECTION_NAME
class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        print("Loading embedding model (first time will download ~80MB)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding model loaded successfully!")
        self.persist_directory = persist_directory
        self.vectorstore = None
    def _clean_documents(self, docs):
        clean = []
        for d in docs:
            if not d or not d.page_content:
                continue
            text = d.page_content.strip()
            if not text or len(text) < 5:
                continue
            clean.append(d)
        removed = len(docs) - len(clean)
        if removed > 0:
            print(f"Removed {removed} empty/invalid chunks")
        return clean
    def create_vectorstore(self, documents):
        clean_docs = self._clean_documents(documents)
        if len(clean_docs) == 0:
            raise ValueError("No valid documents to index!")
        self.vectorstore = Chroma.from_documents(
            documents=clean_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=COLLECTION_NAME,
        )
        print(f"Vector store created with {len(clean_docs)} chunks")
        return self.vectorstore
    def load_vectorstore(self):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        print("Vector store loaded successfully")
        return self.vectorstore
    def add_documents(self, documents):
        if self.vectorstore is None:
            self.load_vectorstore()
        clean_docs = self._clean_documents(documents)
        if len(clean_docs) == 0:
            print("No valid new chunks to add, skipping.")
            return
        self.vectorstore.add_documents(clean_docs)
        print(f"Added {len(clean_docs)} clean chunks")

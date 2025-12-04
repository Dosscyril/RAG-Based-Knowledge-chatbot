from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
from backend.document_processor import DocumentProcessor
from backend.embeddings import VectorStore
from backend.retriever import KnowledgeAssistant
app = FastAPI(title="Personal Knowledge Assistant")
app.mount("/static", StaticFiles(directory="frontend"), name="static")
@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
doc_processor = DocumentProcessor()
vector_store = VectorStore()
assistant = None
class Query(BaseModel):
    question: str
    k: int = 4
@app.on_event("startup")
async def startup_event():
    global assistant
    try:
        vs = vector_store.load_vectorstore()
        assistant = KnowledgeAssistant(vs)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"No existing vector store found: {e}")
@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    global assistant
    uploaded_files = []
    for file in files:
        file_path = f"./data/documents/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file_path)
    all_chunks = doc_processor.process_documents(uploaded_files)
    if assistant is None:
        vs = vector_store.create_vectorstore(all_chunks)
        assistant = KnowledgeAssistant(vs)
    else:
        vector_store.add_documents(all_chunks)

    return {
        "message": f"Successfully processed {len(files)} files",
        "chunks_created": len(all_chunks),
    }
@app.post("/query")
async def query_knowledge_base(query: Query):
    if assistant is None:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload documents first.",
        )

    try:
        result = assistant.query(query.question, k=query.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
async def health_check():
    return {"status": "healthy", "has_documents": assistant is not None}
if __name__ == "__main__":
    print("🌐 Frontend available at: http://localhost:8000")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

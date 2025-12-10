import os
import streamlit as st

from backend.config import CHROMA_DIR
from backend.document_processor import DocumentProcessor
from backend.embeddings import VectorStore
from backend.retriever import KnowledgeAssistant
st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)
st.markdown("""
<style>
.source-box {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    padding: 8px 10px;
    margin-top: 6px;
    font-size: 13px;
    background-color: #f9fafb;
}
.source-filename {
    font-weight: 600;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(persist_directory=CHROMA_DIR)

if "assistant" not in st.session_state:
    try:
        vs = st.session_state.vector_store.load_vectorstore()
        st.session_state.assistant = KnowledgeAssistant(vs)
        st.session_state.has_documents = True
    except:
        st.session_state.assistant = None
        st.session_state.has_documents = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
#side bar 
st.sidebar.title("📁 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / TXT files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if st.sidebar.button("Upload & Process", use_container_width=True):
    if not uploaded_files:
        st.sidebar.error("Please select files before uploading.")
    else:
        save_dir = "data/documents"
        os.makedirs(save_dir, exist_ok=True)

        saved_paths = []
        for uf in uploaded_files:
            save_path = os.path.join(save_dir, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(save_path)

        with st.spinner("Processing documents..."):
            dp = st.session_state.doc_processor
            chunks = dp.process_documents(saved_paths)

            vs = st.session_state.vector_store

            if st.session_state.assistant is None:
                store = vs.create_vectorstore(chunks)
                st.session_state.assistant = KnowledgeAssistant(store)
            else:
                vs.add_documents(chunks)

            st.session_state.chunks_count += len(chunks)
            st.session_state.has_documents = True

        st.sidebar.success(f"Processed {len(uploaded_files)} files")


st.sidebar.markdown("---")
st.sidebar.subheader("📊 Status")

if st.session_state.has_documents:
    st.sidebar.success("Documents Ready")
else:
    st.sidebar.error("No Documents Yet")

st.sidebar.write(f"Chunks Processed: **{st.session_state.chunks_count}**")

#chat area
st.title("🧠 Personal Knowledge Assistant")
st.caption("Upload documents on the left & ask questions below.")

chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("No messages yet. Upload documents and ask something!")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                sources = msg.get("sources", [])
                if sources:
                    st.markdown("**📚 Sources:**")
                    for src in sources:
                        filename = src.get("filename", "Unknown")
                        snippet = src["content"][:200] + "..."
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-filename">{filename}</div>
                                <div>{snippet}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
#chat input 
query = st.chat_input("Ask a question about your documents...")

if query:
    if not st.session_state.has_documents or st.session_state.assistant is None:
        with st.chat_message("assistant"):
            st.error("Please upload documents first.")
    else:
        # store user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant = st.session_state.assistant
                try:
                    result = assistant.query(query, k=4)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    sources = []

                st.markdown(answer)

                # show sources under assistant message
                if sources:
                    st.markdown("**📚 Sources:**")
                    for src in sources:
                        filename = src.get("filename", "Unknown")
                        snippet = src["content"][:200] + "..."
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-filename">{filename}</div>
                                <div>{snippet}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # store assistant reply
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        st.rerun()

import os
import re
import docx2txt
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
SECTION_PATTERNS = [
    r"^[0-9]+\.\s",
    r"^[0-9]+\.[0-9]+\s",
    r"^CHAPTER\s+[0-9IVX]+",
    r"^UNIT\s*[-–]\s*[IVX]+",
    r"^[A-Z][A-Z ]{4,}$",
    r"^#+\s+[A-Za-z].*",
]
_HEADING_RE = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)
        text = text.replace("\x00", "")
        text = re.sub(r"<.*?>", "", text)
        text = text.encode("utf-8", "ignore").decode()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 20:
            return ""
        return text
    def split_into_sections(self, text: str) -> List[str]:
        lines = text.split("\n")
        sections = []
        current = []
        for line in lines:
            if _HEADING_RE.match(line.strip()):
                if current:
                    sections.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append("\n".join(current).strip())
        return sections
    def split_section_into_chunks(self, section_text: str) -> List[Document]:
        sentences = [s.strip() for s in _SENTENCE_RE.split(section_text) if s.strip()]
        chunks = []
        cur = []
        cur_len = 0
        for s in sentences:
            s = self.clean_text(s)
            if not s:
                continue
            slen = len(s)
            if cur_len + slen > CHUNK_SIZE and cur:
                text = " ".join(cur)
                cleaned = self.clean_text(text)
                if cleaned:
                    chunks.append(Document(page_content=cleaned))
                overlap = cleaned[-CHUNK_OVERLAP:]
                overlap = self.clean_text(overlap)
                cur = [overlap] if overlap else []
                cur_len = len(overlap)
            cur.append(s)
            cur_len += slen + 1
        if cur:
            text = " ".join(cur)
            cleaned = self.clean_text(text)
            if cleaned:
                chunks.append(Document(page_content=cleaned))
        return chunks
    def load_file(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            pages = PyPDFLoader(file_path).load()
            text = "\n".join([p.page_content for p in pages])
        elif ext == ".txt":
            docs = TextLoader(file_path, encoding="utf-8").load()
            text = docs[0].page_content
        elif ext == ".docx":
            text = docx2txt.process(file_path) or ""
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        text = self.clean_text(text)
        if not text:
            return []
        sections = self.split_into_sections(text)
        all_chunks = []
        for sec in sections:
            all_chunks.extend(self.split_section_into_chunks(sec))
        filename = os.path.basename(file_path)
        for idx, c in enumerate(all_chunks):
            c.metadata["chunk_id"] = idx
            c.metadata["source"] = filename
        return all_chunks
    def process_documents(self, file_list: List[str]) -> List[Document]:
        chunks = []
        for f in file_list:
            chunks.extend(self.load_file(f))
        return chunks

"""
Core retrieval-augmented generation utilities for the HR Policy Neural Search app.
Separates ingestion, chunking, embedding, vector store persistence, and retrieval chain creation.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an HR Assistant. Answer the user's question based ONLY on the provided context. "
    "If the answer is not in the context, say 'I cannot find that information in the policy documents.' "
    "Do not hallucinate outside info. Keep answers professional and concise. "
    "Always cite the relevant policy section or document name in your answer."
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PDF Ingestion
# ---------------------------------------------------------------------------


def load_pdfs(uploaded_files: Iterable) -> List[Document]:
    """
    Convert uploaded PDF files (Streamlit UploadedFile or file-like) into LangChain Documents.
    Metadata includes the original filename for later citation.
    """
    documents: List[Document] = []
    for uploaded in uploaded_files:
        file_bytes = uploaded.read()
        reader = PdfReader(io.BytesIO(file_bytes))
        source_name = getattr(uploaded, "name", "uploaded.pdf")
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():  # Only add non-empty pages
                doc = Document(
                    page_content=text,
                    metadata={"source": source_name, "page": page_number},
                )
                documents.append(doc)
        # Reset file pointer for potential re-read
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
    return documents


def load_pdf_from_path(pdf_path: Path) -> List[Document]:
    """Load a PDF from a filesystem path."""
    if not pdf_path.exists():
        return []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        documents: List[Document] = []
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_number},
                )
                documents.append(doc)
        return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def split_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents using semantic overlap to preserve context."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Embeddings & Vector Store
# ---------------------------------------------------------------------------


def build_embeddings() -> HuggingFaceEmbeddings:
    """Return a CPU-friendly embedding model."""
    import nest_asyncio
    try:
        nest_asyncio.apply()
    except Exception:
        pass
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(documents: List[Document], persist_dir: str) -> FAISS:
    """Create and persist a FAISS index from documents."""
    _ensure_dir(Path(persist_dir))
    embeddings = build_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore


def load_vectorstore(persist_dir: str) -> Optional[FAISS]:
    """Load an existing FAISS index from disk if present."""
    index_path = Path(persist_dir) / "index.faiss"
    if not index_path.exists():
        return None
    embeddings = build_embeddings()
    return FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def upsert_vectorstore(documents: List[Document], persist_dir: str) -> FAISS:
    """
    Build a new index if none exists; otherwise merge new docs into the current index.
    """
    if not documents:
        existing = load_vectorstore(persist_dir)
        if existing:
            return existing
        raise ValueError("No documents provided and no existing index found.")

    existing = load_vectorstore(persist_dir)
    if existing is None:
        return create_vectorstore(documents, persist_dir)
    # Add new documents and re-persist
    existing.add_documents(documents)
    existing.save_local(persist_dir)
    return existing


def clear_vectorstore(persist_dir: str) -> bool:
    """Remove the persisted FAISS index directory."""
    path = Path(persist_dir)
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def get_document_count(vectorstore: FAISS) -> int:
    """Return the number of vectors in the store."""
    try:
        return vectorstore.index.ntotal
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def build_retriever(vectorstore: FAISS, *, k: int = 4):
    """Return a retriever configured for top-k similarity search."""
    return vectorstore.as_retriever(search_kwargs={"k": min(k, 3)})


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_llm(
    model: str = "models/gemini-2.0-flash",
    temperature: float = 0.1,
):
    """
    Instantiate the Google Gemini chat model.
    Uses gemini-2.0-flash as default (fast and capable). Override via GEMINI_MODEL env var.
    Available models: models/gemini-2.0-flash, models/gemini-2.5-flash, models/gemini-2.5-pro
    """
    model_name = os.getenv("GEMINI_MODEL", model)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it before running the application."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        max_output_tokens=256,
        request_timeout=30,
        max_retries=1,  # fail fast to avoid long waits on rate limits
    )


# ---------------------------------------------------------------------------
# QA Chain
# ---------------------------------------------------------------------------


def build_answer_prompt() -> PromptTemplate:
    """Prompt injecting the system instruction and retrieved context."""
    template = f"""{SYSTEM_PROMPT}

Context from HR Policy Documents:
{{context}}

Question: {{question}}

Instructions: Answer the question based on the context above. Cite the specific policy section or document name. If the information is not in the context, clearly state that.

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


def build_qa_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """
    Create a conversational retrieval chain that returns both the answer and source documents.
    """
    llm = build_llm()
    retriever = build_retriever(vectorstore)
    prompt = build_answer_prompt()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )
    return chain


def qa_with_history(
    chain: ConversationalRetrievalChain,
    question: str,
    history: List,
) -> dict:
    """
    Run the QA chain with limited chat history.
    Expects `history` as a list of (human, ai) tuple pairs or LangChain messages.
    """
    chat_history = _normalize_history(history)
    try:
        return chain.invoke({"question": question, "chat_history": chat_history})
    except Exception as e:
        # Return a structured error response
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}",
            "source_documents": [],
        }


def _normalize_history(history: List) -> List[Tuple[str, str]]:
    """
    Convert varied history representations to list of (human_msg, ai_msg) tuples
    expected by ConversationalRetrievalChain.
    """
    normalized: List[Tuple[str, str]] = []

    # Group consecutive human/ai messages into pairs
    human_msg = None
    for item in history or []:
        if isinstance(item, tuple) and len(item) == 2:
            # Already a tuple pair
            normalized.append(item)
        elif isinstance(item, HumanMessage):
            human_msg = item.content
        elif isinstance(item, AIMessage):
            if human_msg is not None:
                normalized.append((human_msg, item.content))
                human_msg = None
        elif isinstance(item, dict) and "role" in item and "content" in item:
            role = item["role"]
            if role == "user":
                human_msg = item["content"]
            elif role == "assistant" and human_msg is not None:
                normalized.append((human_msg, item["content"]))
                human_msg = None

    return normalized


# ---------------------------------------------------------------------------
# Citation Helpers
# ---------------------------------------------------------------------------


def format_source_attribution(doc: Document) -> str:
    """Human-friendly source string for citations."""
    source = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page")
    if page is not None:
        page_num = page + 1 if isinstance(page, int) else page
        return f"{source} (Page {page_num})"
    return source


def get_unique_sources(docs: List[Document]) -> List[str]:
    """Get unique source names from a list of documents."""
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources

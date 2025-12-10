"""
HR Policy Neural Search - Streamlit Chat Interface
A RAG-powered chatbot for querying HR policy documents.
"""

import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import rag_engine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Neural Search",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

PERSIST_DIR = "faiss_store"
SAMPLE_PDF_PATH = Path("sample_data/Acme_Corp_HR_Policy_Handbook.pdf")
MAX_HISTORY_TURNS = 3

# ---------------------------------------------------------------------------
# Custom CSS - Professional Dark Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling - professional dark */
    [data-testid="stSidebar"] {
        background: #1e2330;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    
    /* Consistent button styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #3b82f6;
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #2563eb;
    }
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background-color: transparent;
        border: 1px solid #475569;
        color: #e2e8f0;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #334155;
        border-color: #64748b;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        background-color: #374151;
        border: 1px solid #4b5563;
        color: #e2e8f0;
    }
    
    .stDownloadButton > button:hover {
        background-color: #4b5563;
        border-color: #6b7280;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    
    /* Alert boxes - consistent styling */
    .stAlert {
        border-radius: 6px;
        border: none;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.875rem;
        color: #94a3b8;
        background: transparent;
        border-radius: 6px;
    }
    
    /* Source citation styling */
    .source-citation {
        background-color: #1e293b;
        border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.875rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Header styling */
    .main-header {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    /* Step cards - professional muted colors */
    .step-card {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        height: 100%;
    }
    
    .step-card h3 {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .step-card p {
        color: #94a3b8;
        font-size: 0.875rem;
        line-height: 1.5;
        margin: 0;
    }
    
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
    }
    
    /* Example question buttons */
    .example-btn {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        text-align: left;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-btn:hover {
        background: #334155;
        border-color: #475569;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3b82f6;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-ready {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-empty {
        background: rgba(100, 116, 139, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 6px;
    }
    
    /* Divider */
    hr {
        border-color: #334155;
        margin: 1rem 0;
    }
    
    /* Caption text */
    .stCaption {
        color: #64748b;
    }
    
    /* Info box */
    .info-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 1rem;
        color: #94a3b8;
        font-size: 0.875rem;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------


def init_state():
    """Initialize all session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[dict] = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files: List[str] = []

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    # Try to load existing vectorstore on startup
    if not st.session_state.vectorstore_ready:
        try:
            vectorstore = rag_engine.load_vectorstore(PERSIST_DIR)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.qa_chain = rag_engine.build_qa_chain(vectorstore)
                st.session_state.vectorstore_ready = True
        except Exception as e:
            st.session_state.last_error = str(e)


def to_langchain_history(history: List[dict]) -> List:
    """Convert session chat history to LangChain message objects (last N turns)."""
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]
    lc_messages = []
    for item in trimmed:
        role = item["role"]
        if role == "user":
            lc_messages.append(HumanMessage(content=item["content"]))
        else:
            lc_messages.append(AIMessage(content=item["content"]))
    return lc_messages


# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------


def process_documents(uploaded_files=None, use_sample: bool = False) -> tuple[bool, str]:
    """Process uploaded PDFs or sample file and update the vector store."""
    try:
        documents = []

        if use_sample:
            if not SAMPLE_PDF_PATH.exists():
                return False, f"Sample file not found at {SAMPLE_PDF_PATH}"
            
            docs = rag_engine.load_pdf_from_path(SAMPLE_PDF_PATH)
            if not docs:
                return False, "Could not extract any content from the sample PDF."
            
            documents.extend(docs)
            if SAMPLE_PDF_PATH.name not in st.session_state.indexed_files:
                st.session_state.indexed_files.append(SAMPLE_PDF_PATH.name)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Reset file pointer before reading
                    uploaded_file.seek(0)
                    docs = rag_engine.load_pdfs([uploaded_file])
                    if docs:
                        documents.extend(docs)
                        if uploaded_file.name not in st.session_state.indexed_files:
                            st.session_state.indexed_files.append(uploaded_file.name)
                except Exception as e:
                    return False, f"Error reading {uploaded_file.name}: {str(e)}"

        if not documents:
            return False, "No content could be extracted from the provided documents."

        chunks = rag_engine.split_documents(documents)
        if not chunks:
            return False, "Document chunking produced no results."

        vectorstore = rag_engine.upsert_vectorstore(chunks, PERSIST_DIR)
        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = rag_engine.build_qa_chain(vectorstore)
        st.session_state.vectorstore_ready = True
        st.session_state.last_error = None

        return True, f"Successfully processed {len(documents)} pages into {len(chunks)} chunks."

    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        st.session_state.last_error = error_detail
        return False, f"Processing failed: {str(e)}"


def clear_index():
    """Clear the vector store and reset state."""
    rag_engine.clear_vectorstore(PERSIST_DIR)
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.vectorstore_ready = False
    st.session_state.indexed_files = []
    st.session_state.chat_history = []
    st.session_state.last_error = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar():
    """Render the sidebar with document management and settings."""
    with st.sidebar:
        st.markdown("## Document Management")
        st.divider()

        # Sample Document Section
        st.markdown("#### Sample HR Policy")

        if SAMPLE_PDF_PATH.exists():
            col1, col2 = st.columns(2)

            with col1:
                with open(SAMPLE_PDF_PATH, "rb") as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=SAMPLE_PDF_PATH.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )

            with col2:
                if st.button(
                    "Load Sample",
                    use_container_width=True,
                    disabled=st.session_state.processing,
                ):
                    st.session_state.processing = True
                    with st.spinner("Processing..."):
                        success, message = process_documents(use_sample=True)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    st.session_state.processing = False
                    if success:
                        st.rerun()

            st.caption("Acme Corp HR Policy Handbook • 7 sections")
        else:
            st.error("Sample file not found.")

        st.divider()

        # Upload Section
        st.markdown("#### Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button(
                "Process Uploads",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            ):
                st.session_state.processing = True
                with st.spinner("Processing documents..."):
                    success, message = process_documents(uploaded_files=uploaded_files)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                st.session_state.processing = False
                if success:
                    st.rerun()

        st.divider()

        # Index Status
        st.markdown("#### Knowledge Base")

        if st.session_state.vectorstore_ready and st.session_state.vectorstore:
            doc_count = rag_engine.get_document_count(st.session_state.vectorstore)
            
            st.markdown(
                """<div class="status-indicator status-ready">
                    <span>●</span> Ready
                </div>""",
                unsafe_allow_html=True,
            )
            
            st.metric("Indexed Chunks", doc_count)

            if st.session_state.indexed_files:
                with st.expander("View indexed files", expanded=False):
                    for fname in st.session_state.indexed_files:
                        st.markdown(f"• {fname}")
        else:
            st.markdown(
                """<div class="status-indicator status-empty">
                    <span>○</span> Empty
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption("Load documents to enable search.")

        st.divider()

        # Actions
        st.markdown("#### Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Clear Chat",
                use_container_width=True,
                disabled=not st.session_state.chat_history,
            ):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button(
                "Reset Index",
                use_container_width=True,
                disabled=not st.session_state.vectorstore_ready,
            ):
                clear_index()
                st.rerun()

        # Export Chat
        if st.session_state.chat_history:
            st.divider()
            st.markdown("#### Export")

            chat_export = {
                "exported_at": datetime.now().isoformat(),
                "messages": st.session_state.chat_history,
            }

            st.download_button(
                label="Export Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Footer
        st.divider()
        st.caption("All data processed locally.")


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------


def render_chat_history():
    """Render the chat message history."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                sources = message["sources"]
                if sources:
                    with st.expander("View sources", expanded=False):
                        for idx, source in enumerate(sources, start=1):
                            st.markdown(f"**{source['citation']}**")
                            st.markdown(
                                f"<div class='source-citation'>{source['content'][:500]}{'...' if len(source['content']) > 500 else ''}</div>",
                                unsafe_allow_html=True,
                            )
                            if idx < len(sources):
                                st.divider()


def handle_user_input(user_input: str):
    """Process user input and generate response."""
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                history = to_langchain_history(st.session_state.chat_history[:-1])
                result = rag_engine.qa_with_history(
                    st.session_state.qa_chain, user_input, history
                )

                answer = result.get("answer", "I couldn't generate an answer.")
                source_docs = result.get("source_documents", [])

                st.markdown(answer)

                sources = []
                if source_docs:
                    with st.expander("View sources", expanded=False):
                        for idx, doc in enumerate(source_docs, start=1):
                            citation = rag_engine.format_source_attribution(doc)
                            content = doc.page_content.strip()

                            sources.append({"citation": citation, "content": content})

                            st.markdown(f"**{citation}**")
                            st.markdown(
                                f"<div class='source-citation'>{content[:500]}{'...' if len(content) > 500 else ''}</div>",
                                unsafe_allow_html=True,
                            )
                            if idx < len(source_docs):
                                st.divider()

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as exc:
                error_msg = f"Error: {str(exc)}"
                if "429" in error_msg or "quota" in error_msg.lower():
                    error_msg = (
                        "The model is rate-limited right now. Please wait a minute and try again, "
                        "or switch to a different Gemini model via GEMINI_MODEL env var."
                    )
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )


def render_welcome_message():
    """Display welcome message and quick start guide."""
    st.markdown(
        """
        <div style='text-align: center; padding: 1.5rem 0;'>
            <h2 style='color: #f1f5f9; font-weight: 600; margin-bottom: 0.5rem;'>Welcome to HR Policy Neural Search</h2>
            <p style='color: #94a3b8; font-size: 1rem;'>
                Ask questions about HR policies and get instant answers with source citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class='step-card'>
                <div class='step-number'>1</div>
                <h3>Load Documents</h3>
                <p>Load the sample HR policy or upload your own PDF documents from the sidebar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='step-card'>
                <div class='step-number'>2</div>
                <h3>Ask Questions</h3>
                <p>Type your questions in natural language using the chat input below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class='step-card'>
                <div class='step-number'>3</div>
                <h3>Get Answers</h3>
                <p>Receive accurate answers with citations to the relevant policy sections.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Example questions
    st.markdown("#### Example Questions")

    example_questions = [
        "How many days per week can I work remotely?",
        "What is the PTO accrual rate?",
        "How does the 401(k) matching work?",
        "What is the grievance process?",
        "What are the travel expense limits?",
    ]

    cols = st.columns(len(example_questions))
    for i, (col, question) in enumerate(zip(cols, example_questions)):
        with col:
            if st.button(
                question[:25] + "..." if len(question) > 25 else question,
                key=f"example_{i}",
                disabled=not st.session_state.vectorstore_ready,
                use_container_width=True,
            ):
                handle_user_input(question)
                st.rerun()


def main():
    """Main application entry point."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_state()

    render_sidebar()

    # Main content area
    st.markdown(
        "<h1 class='main-header'>📋 HR Policy Neural Search</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "AI-powered search for HR policy documents • Answers grounded in your documents • Source citations included"
    )

    st.divider()

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "**GOOGLE_API_KEY not set.** Please set the environment variable before running the app."
        )
        st.code("export GOOGLE_API_KEY='your-api-key-here'", language="bash")
        st.stop()

    # Show welcome if no documents indexed
    if not st.session_state.vectorstore_ready:
        render_welcome_message()
        st.markdown(
            """<div class='info-box'>
                <strong>Getting Started:</strong> Load the sample document or upload your own PDFs using the sidebar to begin.
            </div>""",
            unsafe_allow_html=True,
        )

    render_chat_history()

    # Chat input
    placeholder_text = "Ask about HR policies..." if st.session_state.vectorstore_ready else "Load documents to start chatting..."
    
    user_input = st.chat_input(
        placeholder_text,
        disabled=not st.session_state.vectorstore_ready,
    )

    if user_input and st.session_state.qa_chain:
        handle_user_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()

# HR Policy Neural Search 📋

A **Retrieval-Augmented Generation (RAG)** application for querying HR policy documents using natural language. Built with Streamlit, LangChain, FAISS, and Google Gemini.

![HR Policy Neural Search](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.x-green.svg)

## ✨ Features

- **📄 PDF Document Ingestion**: Upload and process multiple HR policy PDFs
- **🔍 Semantic Search**: Find relevant information using natural language queries
- **💬 Conversational Interface**: WhatsApp-style chat with conversation history
- **📚 Source Citations**: Every answer includes references to source documents
- **📥 Sample Document**: Pre-loaded HR policy handbook for immediate testing
- **💾 Persistent Index**: FAISS vector store saved to disk for fast reloads
- **📤 Chat Export**: Download conversation history as JSON
- **🎨 Modern UI**: Beautiful gradient-styled interface with dark theme

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF Upload    │────▶│  Text Extraction │────▶│    Chunking     │
│   (PyPDF)       │     │   (Per Page)     │     │ (1000/200 char) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Google Gemini  │◀────│   Retrieval      │◀────│  FAISS Index    │
│  (LLM Answer)   │     │   (Top-K)        │     │  (Embeddings)   │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Chat Interface                      │
│  • User Question → Context Retrieval → LLM Answer + Citations   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google API Key with Gemini API access

### Installation

1. **Clone the repository**
   ```bash
   cd "HR Policy Neural Search"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Google API Key**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
HR Policy Neural Search/
├── app.py                 # Streamlit UI and main application
├── rag_engine.py          # RAG pipeline: ingestion, embedding, retrieval
├── requirements.txt       # Python dependencies
├── sample_data/           # Sample HR policy documents
│   └── Acme_Corp_HR_Policy_Handbook.pdf
├── faiss_store/           # Persisted vector index (auto-created)
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `models/gemini-2.0-flash` |

### Chunking Parameters

Edit `rag_engine.py` to adjust:
- `CHUNK_SIZE`: Characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## 💡 Usage

### Loading Documents

1. **Sample Document**: Click "Load Sample" in the sidebar to load the pre-included HR policy
2. **Upload Your Own**: Drag & drop or browse to upload PDF files
3. **Download Sample**: Click "Download" to get the sample PDF for reference

### Asking Questions

Once documents are loaded:
- Type your question in the chat input
- Click example questions for quick testing
- View source citations in the expandable "View Source Context" section

### Example Questions

- "How many days per week can I work remotely?"
- "What is the PTO accrual rate for new employees?"
- "How does the 401(k) matching work?"
- "What is the process for filing a grievance?"
- "What are the travel expense limits?"

## 🛠️ Technical Details

### RAG Pipeline

1. **Ingestion**: PDFs are parsed page-by-page using PyPDF
2. **Chunking**: `RecursiveCharacterTextSplitter` with semantic separators
3. **Embedding**: `all-MiniLM-L6-v2` from SentenceTransformers (runs on CPU)
4. **Storage**: FAISS vector store with persistence to disk
5. **Retrieval**: Top-4 similarity search
6. **Generation**: Google Gemini with grounded prompting

### System Prompt

The LLM is instructed to:
- Answer ONLY from provided context
- Cite specific policy sections
- Acknowledge when information is not found
- Maintain professional, concise responses

## 🔒 Privacy

- All document processing happens locally
- Embeddings are computed on your machine
- Only the query and retrieved context are sent to Google Gemini
- No documents are stored externally

## 📝 License

MIT License - feel free to use and modify for your needs.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- New features
- Documentation improvements

---

Built with ❤️ using Streamlit, LangChain, and Google Gemini


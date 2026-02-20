# Changelog — Compatibility Fixes for LangChain 1.x, llama_index 0.10+, ChromaDB 1.x, OpenAI API 1.x

## 2026-02-19 — LangChain 1.x + ChromaDB Compatibility Fix

### Problem
Running the original script produced:

```
AttributeError: 'NoneType' object has no attribute 'info'
```

This was caused by two separate issues: broken import paths from the LangChain 0.x → 1.x migration, and Python 3.14 being incompatible with the pydantic v1 compatibility layer used internally by ChromaDB.

---

### 1. Import Path Migrations

LangChain 1.x split its monolithic package into several smaller packages. The old import paths no longer exist.

| Old (broken) | New |
|---|---|
| `from langchain.document_loaders import DirectoryLoader, TextLoader` | `from langchain_community.document_loaders import DirectoryLoader, TextLoader` |
| `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings, OpenAI` |
| `from langchain import OpenAI` | merged into `langchain_openai` line above |
| `from langchain.vectorstores import Chroma` | `from langchain_community.vectorstores import Chroma` |
| `from langchain.text_splitter import CharacterTextSplitter` | `from langchain_text_splitters import CharacterTextSplitter` |

---

### 2. Removed `RetrievalQA` Chain

`RetrievalQA` was removed in LangChain 1.x.

**Before:**
```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

def query(q):
    print(qa.run(q))
```

**After** — replaced with direct retriever + LLM invocation:
```python
retriever = docsearch.as_retriever()
llm = OpenAI()

def query(q):
    print("Query: ", q)
    docs = retriever.invoke(q)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = llm.invoke(
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
    )
    print("Answer: ", answer)
```

---

### 3. Python Version Requirement

ChromaDB internally uses `pydantic.v1.BaseSettings`. Pydantic explicitly dropped support for this compatibility shim on Python 3.14+, causing a hard crash at import time.

**Requirement:** Python 3.12 (or 3.11/3.13 — anything below 3.14).

Create a fresh conda environment:
```bash
conda create -n llm-env python=3.12
conda activate llm-env
```

---

### 4. Required pip Packages

The original code assumed LangChain 0.x's all-in-one `langchain` package. With the split into smaller packages, the following must be installed explicitly:

```bash
pip install langchain-community langchain-openai langchain-text-splitters chromadb python-dotenv
```

| Package | Purpose |
|---|---|
| `langchain-community` | `DirectoryLoader`, `TextLoader`, `Chroma` |
| `langchain-openai` | `OpenAIEmbeddings`, `OpenAI` |
| `langchain-text-splitters` | `CharacterTextSplitter` |
| `chromadb` | Vector store backend |
| `python-dotenv` | `.env` file loading for `OPENAI_API_KEY` |

---

## 2026-02-20 — Compatibility Fixes for Current Package Versions

### Files Fixed

**02b_llama_chroma.py** — ChromaDB 1.x
- `chromadb.Client()` → `chromadb.PersistentClient(path="./storage")`
- `create_collection()` → `get_or_create_collection()`

**03_db.py** — LangChain 1.x + Tool-calling
- Changed `OpenAI` → `ChatOpenAI` (required for tool-calling agents)
- `SQLDatabaseChain` → `create_sql_agent(agent_type="openai-tools")`
- `.run()` → `.invoke({"input": query})`

**04_csv.py** — LangChain 1.x + Security
- Changed `OpenAI` → `ChatOpenAI`
- Added `allow_dangerous_code=True` (required in LangChain 1.x for code execution)

**07_custom.py** — llama_index 0.10+ + CPU compatibility
- `max_input_size` → `context_window`
- `max_chunk_overlap` → `chunk_overlap_ratio`
- Removed `device="cuda:0"` for cross-platform compatibility

**11_worldbuilding.py** — Cohere Generate API → Chat API
- Rewrote `generate()` function to use `co.chat()` (Generate API deprecated Sept 15, 2025)
- Model updated to `command-a-03-2025`
- Simplified prompts for Chat API compatibility
- Added error handling for inconsistent outputs
- Note: Token likelihoods not available in Chat API

**13_caching_sqlite.py** — LangChain 1.x legacy
- `from langchain.chains.summarize` → `from langchain_classic.chains.summarize`

**14_streamlit.py** — LangChain 1.x legacy
- `from langchain.agents` → `from langchain_classic.agents`

**15_sql.py** — Groq model update
- `llama3-70b-8192` → `llama-3.3-70b-versatile` (old model decommissioned)

**16_repl.py** — Groq model update
- `llama3-70b-8192` → `llama-3.3-70b-versatile` (old model decommissioned)

**requirements.txt** — Cross-platform + new packages
- Commented out `triton` and `uvloop` (Linux-only packages)
- Added `openai-agents==0.9.2` (for files 19-24)
- Added `langchain-groq==1.1.2` (for 15_sql.py, 16_repl.py)
- Added `langchain-classic==0.3.21` (for legacy APIs)
- Updated all package versions to tested environment

### Key Package Versions
- langchain==1.2.10
- langchain-community==0.4.1
- langchain-openai==1.1.10
- chromadb==1.5.0
- llama-index==0.14.15
- Python 3.12 required

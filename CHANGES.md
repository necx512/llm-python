# Changelog — `01_qna.py`

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

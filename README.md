# 📄 Intelligent Document Analyzer  
### Retrieval-Augmented Generation (RAG) System for Document Q&A

## 🔍 Overview

The **Intelligent Document Analyzer** is an AI-powered system that allows users to **upload documents (PDF/TXT)** and ask **natural-language questions** about their content.

Instead of manually searching through documents, users can interact with them conversationally. The system retrieves the most relevant document sections and generates **grounded, context-aware answers** using a **Retrieval-Augmented Generation (RAG)** pipeline.

This project demonstrates how **Large Language Models (LLMs)** can be safely combined with retrieval systems to analyze unstructured data such as policies, reports, and technical documentation.

---

## ✨ Key Features

- 📂 Upload PDF or TXT documents via a web interface  
- 🔍 Hybrid retrieval using **BM25 (keyword)** + **vector similarity search**  
- 🎯 Cross-encoder reranking for improved answer relevance  
- 🧠 LLM-based answer generation grounded strictly in document context  
- 🚫 Hallucination control — answers are generated only from retrieved content  
- 🌐 Interactive **Streamlit** web application  
- 💻 Runs locally with **open-source models** (no API keys required)

---

## 🧠 How the System Works

### High-Level Architecture

User Uploads Document
↓
Document Chunking
↓
Embeddings + Vector Database
↓
Hybrid Retrieval (BM25 + Vector Search)
↓
Cross-Encoder Reranking
↓
LLM Generates Grounded Answer



---

## ⚙️ Technical Components

### 1️⃣ Document Ingestion
- Supports **PDF and TXT files**
- Documents are loaded and split into semantically meaningful chunks

### 2️⃣ Chunking Strategy
- Optimized for factual and structured documents
- Preserves headings, bullet points, and short sections

### 3️⃣ Embeddings & Vector Store
- Sentence-level embeddings represent document chunks
- Vectors are stored locally in a vector database for efficient similarity search

### 4️⃣ Hybrid Retrieval
- **BM25** captures exact keyword matches  
- **Vector Search** captures semantic meaning  
- Results are combined to improve recall and precision

### 5️⃣ Reranking
- A cross-encoder model reranks retrieved chunks
- Ensures only the most relevant context is passed to the LLM

### 6️⃣ Answer Generation (RAG)
- An instruction-tuned LLM generates answers
- The model is restricted to retrieved context
- If the answer is not present, it responds safely  
  (e.g., *"Not mentioned in the document"*).

---

## 📑 Suitable Document Types

This system performs best on:

- ✅ Policies and SOPs  
- ✅ Technical documentation  
- ✅ Interview Q&A documents  
- ✅ Reports and structured notes  
- ⚠️ Resumes (fact-based questions only)

It is **not designed** for deep reasoning, evaluation, or opinion-based tasks.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/intelligent-document-analyzer.git
cd intelligent-document-analyzer


## 🧭 How to Use the Application

1. Upload one or more **PDF/TXT** files  
2. Ask **fact-based questions** about the document  
3. View generated answers along with the **retrieved context**

---

## 🧪 Example Questions

### ✅ Recommended Questions
- What is Docker and why is it used?
- What tools are mentioned in the document?
- What is the purpose of Docker Compose?
- Which commands are listed for container management?

### ❌ Avoid Asking
- Is this document good?
- Evaluate the candidate
- Which option is better?

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** – Web UI  
- **LangChain (LCEL)** – Prompt + LLM composition  
- **Hugging Face Transformers** – LLM inference  
- **Sentence Transformers** – Embeddings  
- **ChromaDB** – Vector storage  
- **rank_bm25** – Keyword retrieval  

> ⚠️ This project uses **open-source models only**.  
> No OpenAI or paid API keys are required.

---

## 🎯 Why This Project Matters

This project demonstrates:

- Practical application of **Retrieval-Augmented Generation (RAG)** architectures  
- Handling and analysis of **unstructured data** (PDFs, text documents)  
- Safe LLM usage with **hallucination control**  
- Hybrid retrieval strategies used in **real-world systems**  
- AI-assisted analytics rather than pure model training  

It reflects **production-style thinking**, not just experimentation.

---

## 📌 Limitations

- Optimized for **fact extraction**, not deep reasoning  
- Best results come from **structured, explicit documents**  
- Not intended for legal, medical, or evaluative decision-making  

---

## 🔮 Future Improvements

- Conversational memory (LCEL-based)
- Citation highlighting with page numbers
- Multi-document comparison
- Stronger local LLMs (Mistral / LLaMA)
- REST API backend (FastAPI)

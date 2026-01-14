import streamlit as st
import tempfile
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.loaders import load_documents
from app.chunking import chunk_documents
from app.embeddings import load_embedding_model
from app.vectordb import create_vector_db
from app.bm25 import create_bm25_retriever
from app.hybrid_retriever import hybrid_search
from app.reranker import rerank

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate



# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Intelligent Document Analytics", layout="wide")
st.title("📄 Intelligent Document Analytics System")

st.markdown(
    "Upload documents and ask questions. "
    "Answers are generated **only from the uploaded documents**."
)

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("⬆️ Upload documents to begin.")
    st.stop()

# ---------------- INITIALIZE RAG ----------------
@st.cache_resource(show_spinner=True)
def initialize_rag(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Load & process documents
        documents = load_documents(temp_dir)
        chunks = chunk_documents(documents)

        # Embeddings + Vector DB
        embedding = load_embedding_model()
        vectordb = create_vector_db(chunks, embedding)

        # Retrievers
        bm25 = create_bm25_retriever(chunks)
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        # LLM (PyTorch only)
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            framework="pt",
            max_new_tokens=256
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        # Prompt (LCEL-compatible)
        prompt = PromptTemplate(
            template="""
You are an intelligent document assistant.
Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )

        # LCEL RAG chain (STABLE)
        rag_chain = prompt | llm

        return bm25, vector_retriever, rag_chain

bm25, vector_retriever, rag_chain = initialize_rag(uploaded_files)

# ---------------- QUERY ----------------
query = st.text_input("Ask a question about your documents")

if query:
    with st.spinner("Thinking..."):
        retrieved_docs = hybrid_search(query, bm25, vector_retriever)
        reranked_docs = rerank(query, retrieved_docs)

        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        answer = rag_chain.invoke({
            "context": context,
            "question": query
        })

    st.subheader("Answer")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        for i, doc in enumerate(reranked_docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content)

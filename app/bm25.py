from langchain_community.retrievers import BM25Retriever

def create_bm25_retriever(chunks):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    return bm25

from langchain_community.vectorstores import Chroma

def create_vector_db(chunks, embedding_model):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    return vectordb

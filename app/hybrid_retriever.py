def hybrid_search(query, bm25, vector_retriever):
    bm25_docs = bm25.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    seen = set()
    combined = []

    for doc in bm25_docs + vector_docs:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)

    return combined

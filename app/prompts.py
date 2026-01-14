from langchain_core.prompts import PromptTemplate
RAG_PROMPT = PromptTemplate(
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

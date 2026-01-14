from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


def create_lcel_rag():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

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

    # LCEL pipeline (THIS IS THE FIX)
    rag_chain = prompt | llm

    return rag_chain

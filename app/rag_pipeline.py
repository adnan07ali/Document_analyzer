from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import LLMChain
from transformers import pipeline

def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)

def create_rag_chain(llm, prompt, memory):
    return LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline

hf_model = hf_pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=hf_model)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in 3 bullet points"
)

chain = prompt | llm

result = chain.invoke({"topic": "Machine Learning"})

print(result)
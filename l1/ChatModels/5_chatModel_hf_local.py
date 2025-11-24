from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os

os.environ["HF_HOME"] = "/home/vidhiajmera/hf_home"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)

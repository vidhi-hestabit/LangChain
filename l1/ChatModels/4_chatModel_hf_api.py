from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")
print("DEBUG KEY:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))


llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)

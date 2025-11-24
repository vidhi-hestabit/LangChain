from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32, api_key=os.getenv("OPEN_API_KEY"))

documents = [
    "Delhi is the capital of India",
    "Lucknow is the capital of UP",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))


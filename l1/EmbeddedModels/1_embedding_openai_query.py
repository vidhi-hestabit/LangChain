from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32, api_key=os.getenv("OPEN_API_KEY"))

result = embedding.embed_query("Delhi is the capital of India!!")

print(str(result))


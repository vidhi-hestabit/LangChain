from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

ENV_PATH = "/home/vidhiajmera/LangChain/l1/.env"
load_dotenv(ENV_PATH)

print("DEBUG KEY:", os.getenv("OPEN_API_KEY"))

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPEN_API_KEY")
)

result = llm.invoke("Tell me a programming joke only 1.")
print(result)

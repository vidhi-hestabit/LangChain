from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")

print("DEBUG KEY:", os.getenv("OPEN_API_KEY"))

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1,
    api_key=os.getenv("OPEN_API_KEY")
)

result = model.invoke("Write 5 line poem on cricket!")
print(result.content)

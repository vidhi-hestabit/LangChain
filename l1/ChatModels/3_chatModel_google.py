from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")

print("DEBUG KEY:", os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
    api_key=os.getenv("GOOGLE_API_KEY")
)
result = model.invoke("Write 5 line poem on cricket!")
print(result.content)

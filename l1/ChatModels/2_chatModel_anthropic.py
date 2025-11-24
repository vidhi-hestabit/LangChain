from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv("/home/vidhiajmera/LangChain/l1/.env")

print("DEBUG KEY:", os.getenv("OPEN_API_KEY"))

model = ChatAnthropic(
    model="claude-3-100k",
    api_key=os.getenv("OPEN_API_KEY")
)

result = model.invoke("Write 5 line poem on cricket!")
print(result.content)

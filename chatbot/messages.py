from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

model=ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about langchain"),
]

result = model.invoke(messages)

AIMessage(AIMessage(content=result.content))

messages.append(AIMessage(content=result.content))

print(messages)


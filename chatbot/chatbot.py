from langchain_community.chat_models import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Load local model (example: phi3)
model = ChatOllama(model="phi3")

# Chat history
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("Enter your prompt (or type 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # Get model response
    result = model.invoke(chat_history)

    # Add AI response to history
    chat_history.append(AIMessage(content=result.content))

    print("Response:", result.content)

# Print final chat history
print(chat_history)

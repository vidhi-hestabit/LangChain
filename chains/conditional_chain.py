# Feedback -- analyse sentiment 
# -- Positive - 
# -- Negative - 
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# ------------------ MODEL ------------------
model = ChatOllama(
    model="phi3",     # OR llama3, mistral, qwen etc.
    temperature=0.7
)

parser = StrOutputParser()

# ------------------ PROMPTS ------------------

short_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short 3-line explanation on: {topic}"
)

long_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a long, detailed, 10-line explanation on: {topic}"
)

# ------------------ CONDITIONS ------------------
# This function decides whether the topic needs a short or long explanation.

def decide_length(input_data):
    topic = input_data["topic"]

    # Condition: small topics get *short explanation*
    if len(topic.split()) <= 2:
        return "short"
    return "long"

# Convert this python function into a Runnable
condition_node = RunnableLambda(decide_length)

# ------------------ BRANCH ------------------
# RunnableBranch = If / Else logic in LangChain

conditional_chain = RunnableBranch(
    (lambda x: x == "short", short_prompt | model | parser),
    (lambda x: x == "long", long_prompt | model | parser),
)

# Final chain: input → condition → branching chain
chain = condition_node | conditional_chain


# ------------------ RUN ------------------
if __name__ == "__main__":
    result = chain.invoke({"topic": "Black Hole"})
    print("\n=== Final Output ===\n")
    print(result)

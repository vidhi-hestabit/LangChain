from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 5 interesting facts about {topic}."
)

# Replace OpenAI with Ollama
model = ChatOllama(
    model="phi3",     # or llama3, mistral, qwen, etc.
    temperature=0.7
)

# Output parser
parser = StrOutputParser()

# Runnable chain
chain = prompt | model | parser

# Invoke the chain
result = chain.invoke({"topic": "cricket"})

print("\nFinal Output:\n", result)

chain.get_graph().print_ascii()
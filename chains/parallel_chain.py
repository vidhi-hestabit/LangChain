# parallel_chain.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# ---------- OLLAMA MODEL ----------
model = ChatOllama(
    model="phi3",     # Make sure phi3 is installed → ollama pull phi3
    temperature=0.7
)

# ---------- PROMPTS ----------
notes_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write detailed study notes on: {topic}",
)

quiz_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a 5-question quiz on the topic: {topic}",
)

merge_prompt = PromptTemplate(
    input_variables=["notes", "quiz"],
    template="""
Combine the following study notes and quiz into a clean single document.

NOTES:
{notes}

QUIZ:
{quiz}
"""
)

parser = StrOutputParser()

# ---------- PARALLEL CHAIN ----------
parallel_chain = RunnableParallel({
    "notes": notes_prompt | model | parser,
    "quiz": quiz_prompt | model | parser,
})

# ---------- MERGING CHAIN ----------
merge_chain = merge_prompt | model | parser

# ---------- FINAL CHAIN ----------
complete_chain = parallel_chain | merge_chain


# ---------- RUN ----------
if __name__ == "__main__":
    result = complete_chain.invoke({"topic": "Black hole"})
    
    print("\n\n===== FINAL MERGED DOCUMENT =====\n")
    print(result)

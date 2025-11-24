from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

# -------- OLLAMA MODEL (Simple) -----------
model = ChatOllama(
    model="phi3",      # CHANGE this to a model you have pulled in Ollama
    temperature=0.7
)

# -------- PROMPTS -----------
report_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on the topic: {topic}",
)

summary_prompt = PromptTemplate(
    input_variables=["report"],
    template="Write a 5-line summary of this text:\n\n{report}",
)

def sequential_chain(topic: str):
    # Step 1 → Generate full report
    report_query = report_prompt.format(topic=topic)
    report_result = model.invoke(report_query)
    report_text = report_result.content

    # Step 2 → Generate summary
    summary_query = summary_prompt.format(report=report_text)
    summary_result = model.invoke(summary_query)
    summary_text = summary_result.content

    return {
        "report": report_text,
        "summary": summary_text
    }


# ---------- RUN ----------
if __name__ == "__main__":
    topic = "Black hole"
    output = sequential_chain(topic)

    print("\n--- Detailed Report ---\n")
    print(output["report"])

    print("\n--- Final Summary ---\n")
    print(output["summary"])

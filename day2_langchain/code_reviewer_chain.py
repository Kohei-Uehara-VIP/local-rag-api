"""
Day 2: LLM orchestration with LangChain.

Builds an LCEL chain (prompt | llm | StrOutputParser() | extract_code) that sends messy
Python code to a local Ollama model acting as a "Senior Python Code Reviewer" and prints
a clean, refactored version that is ready to save or run.

Usage (from the project root, with the `local-rag-api` conda env activated):
    python day2_langchain/code_reviewer_chain.py
"""

import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

MODEL = "llama3.2:3b"

# Deliberately messy input: poor names, manual index loops, no spacing.
# Expected behavior: sum of squares of even numbers -> 4 + 16 + 36 = 56
MESSY_CODE = """
def calc(l):
    r=[]
    for i in range(0,len(l)):
        if l[i]%2==0:
            r.append(l[i]*l[i])
    t=0
    for x in r: t=t+x
    return t
print(calc([1,2,3,4,5,6]))
"""

# The persona goes in the system message, where chat models weight instructions most.
SYSTEM_PROMPT = """You are a Senior Python Code Reviewer.
Refactor the user's code so that it is clean, readable and Pythonic (PEP 8).
Rules:
- Keep the original behavior exactly the same.
- Use descriptive names, type hints and a short docstring.
- Return ONLY the refactored code in one Python code block, with no explanation."""

# Matches a Markdown code fence such as ```python ... ``` and captures the code inside.
CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the code inside the first Markdown code fence, or the whole text if none."""
    match = CODE_FENCE_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def build_chain():
    """Create the prompt | llm | parser | extract_code chain."""
    # 1) Prompt: persona (system) + a {code} placeholder filled in at runtime (human).
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Refactor this Python code:\n\n{code}"),
        ]
    )

    # 2) Model: the local Ollama model wrapped as a LangChain chat model.
    #    temperature=0 makes the output as deterministic as possible, which suits code.
    llm = ChatOllama(model=MODEL, temperature=0)

    # 3) Parser: converts the model's AIMessage object into a plain string.
    parser = StrOutputParser()

    # 4) Post-processing: strip the Markdown fence so the result is runnable Python.
    #    RunnableLambda turns a normal function into a chain component.
    clean_code = RunnableLambda(extract_code)

    # LCEL: the | operator feeds each component's output into the next one.
    return prompt | llm | parser | clean_code


def main() -> None:
    chain = build_chain()

    print("=== Input: messy code ===")
    print(MESSY_CODE)

    start = time.perf_counter()
    # The dict keys must match the placeholders in the prompt ({code}).
    refactored = chain.invoke({"code": MESSY_CODE})
    elapsed = time.perf_counter() - start

    print(f"=== Output: refactored by {MODEL} ({elapsed:.1f} s) ===")
    print(refactored)


if __name__ == "__main__":
    main()
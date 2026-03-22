from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
# Chains: Chains define sequences of actions, where each step can involve querying an LLM, manipulating data or interacting with external tools. There are two types:

Simple Chains: A single LLM invocation.
Multi-step Chains: Multiple LLMs or actions combined, where each step can take the output from the previous step.

# Prompt Management: LangChain facilitates managing and customizing prompts passed to the LLM. Developers can use PromptTemplates to define how inputs and outputs are formatted before being passed to the model. It also simplifies tasks like handling dynamic variables and prompt engineering, making it easier to control the LLM's behavior.

# Agents: Agents are autonomous systems within LangChain that take actions based on input data. They can call external APIs or query databases dynamically, making decisions based on the situation. These agents leverage LLMs for decision-making, allowing them to respond intelligently to changing input.

# Vector Database: LangChain integrates with a vector database which is used to store and search high-dimensional vector representations of data. This is important for performing similarity searches, where the LLM converts a query into a vector and compares it against the vectors in the database to retrieve relevant information.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=400,
    chunk_overlap=0
)

chunk = splitter.split_text(text)

print(len(chunk))

print(chunk[2])
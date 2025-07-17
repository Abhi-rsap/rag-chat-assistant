from langchain_ollama.llms import OllamaLLM
from vector import retriever
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType


model = OllamaLLM(
    model = "llama3.2"
)


@tool
def search_reviews(question: str) -> str:
    """Useful for answering questions about restaurants using RAG."""
    return retriever.invoke(question)


tools = [search_reviews]
agent_executor = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


question = "What are the best restaurants in Bangalore?"
result = agent_executor.invoke({"input":  question})

print(result['output'])
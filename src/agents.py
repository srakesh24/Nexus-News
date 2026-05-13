import operator
from typing import Annotated,List,TypedDict
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph import StateGraph,END
from langchain_ollama import ChatOllama
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage],operator.add]
    research_data:List[str]
    chart_data:List[dict]
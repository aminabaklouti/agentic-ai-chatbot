# graph_workflow.py
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY") 
os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")

# Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query Arxiv papers")

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia, description="Query Wikipedia")

tavily = TavilySearch()

tools = [arxiv, wikipedia, tavily]

# LLM
llm = ChatGroq(model="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")  # After tools, go back to LLM for final response
graph = builder.compile()

def invoke_graph(user_query: str):
    from langchain.schema import HumanMessage
    messages = graph.invoke({"messages": HumanMessage(content=user_query)})
    
    # Extract assistant response and tools used
    assistant_response = ""
    tools_used = []
    
    for msg in messages['messages']:
        # Get the final assistant message (response)
        if hasattr(msg, 'content') and hasattr(msg, '__class__'):
            if 'AI' in str(type(msg)) and msg.content:
                assistant_response = msg.content
        
        # Check for tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if hasattr(tool_call, 'name'):
                    tools_used.append(tool_call.name)
                elif isinstance(tool_call, dict) and 'name' in tool_call:
                    tools_used.append(tool_call['name'])
    
    # Remove duplicates while preserving order
    tools_used = list(dict.fromkeys(tools_used))
    
    return {
        "response": assistant_response,
        "tools_used": tools_used
    }
import os
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in environment variables")

# Initialize LLM
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),
    ("user", "{text}")
])
parser = StrOutputParser()

# Define LangGraph State
class TranslationState(BaseModel):
    language: str
    text: str
    prompted: Any = None
    response: Any = None
    final_output: str = None

# Define LangGraph Nodes
def prompt_node(state: TranslationState) -> dict:
    # Correctly format as list of chat messages
    prompted = prompt.format_messages(language=state.language, text=state.text)
    return {"prompted": prompted}

def model_node(state: TranslationState) -> dict:
    response = model.invoke(state.prompted)
    return {"response": response}

def parse_node(state: TranslationState) -> dict:
    final_output = parser.invoke(state.response)
    return {"final_output": final_output}

# Build LangGraph
builder = StateGraph(state_schema=TranslationState)
builder.add_node("prompt_node", prompt_node)
builder.add_node("model_node", model_node)
builder.add_node("parse_node", parse_node)

builder.set_entry_point("prompt_node")
builder.add_edge("prompt_node", "model_node")
builder.add_edge("model_node", "parse_node")
builder.add_edge("parse_node", END)

graph = builder.compile()

# FastAPI App
app = FastAPI(title="LangGraph Translation API")
add_routes(app, graph, path="/graph")

# Optional: run graph directly from script
if __name__ == "__main__":
    # Direct test
    input_data = {
        "language": "Hindi",
        "text": "Good morning"
    }
    result = graph.invoke(input_data)
    print("Translation:", result["final_output"])

    # Optional: run FastAPI server directly
    import uvicorn
    uvicorn.run("serve:app", host="localhost", port=8000, reload=True)

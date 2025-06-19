from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
#LLM chatmodel prompt template
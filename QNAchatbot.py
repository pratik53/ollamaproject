import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
import bs4
user_agent = os.getenv("USER_AGENT")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it", groq_api_key = groq_api_key)

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

loader = WebBaseLoader(
    web_path=("https://mahakosh.gov.in"),
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            class_=("post-content",'post-title','post-header')
        )

    )
)

docs = loader.load()
print(docs)

#langchain and openAI
#RAG - Retrieval Augmented Generation

###Steps###
'''
1. Loading dataset like image pdf, text Data Ingestion
2. Split into text chunks
3. Embedding - converting text into vectors
4. Store it into Vector DB like chromadb, FAISS, astradb
'''
#local module#
import json
import requests

##Loader##
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader

##Text chunks##
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter

charater_text_splitter = CharacterTextSplitter(separator = "\n\n", chunk_size = 500, chunk_overlap = 50)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1")])
json_splitter = RecursiveJsonSplitter(max_chunk_size=10)

html_string = """<!DOCTYPE html>
<html>
<body>
<h1>My First Heading</h1>
<p>My first paragraph.</p>
</body>
</html>
"""

#URL splitter#
url = "https://epfigms.gov.in"
final_documents = html_splitter.split_text_from_url(url)

#Html splitter#
final_documents = html_splitter.split_text(html_string)

#text loader#
loader = TextLoader('speech.txt')
text_loader = loader.load()
final_documents = text_splitter.split_documents(text_loader)
character_final_documents = charater_text_splitter.split_documents(text_loader)

#PDF loader#
loader = PyPDFLoader('basic-text.pdf')
pdf_loader = loader.load()
final_documents = text_splitter.split_documents(pdf_loader)
character_final_documents = charater_text_splitter.split_documents(pdf_loader)

#Web based loader#
loader = WebBaseLoader(web_paths=('https://www.moneycontrol.com',),bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_ = ('clearfix MT10'))))
web_loader = loader.load()
final_documents = text_splitter.split_documents(web_loader)
character_final_documents = charater_text_splitter.split_documents(web_loader)

#Arxiv#
'''loader = ArxivLoader(query="1706.03762",load_max_docs=2)
arxiv_loader = loader.load()
final_documents = text_splitter.split_documents(arxiv_loader)
character_final_documents = charater_text_splitter.split_documents(arxiv_loader)

#WikiPedia#
loader = WikipediaLoader(query="Generative AI", load_max_docs=2)
wikipedia_loader = loader.load()
final_documents = text_splitter.split_documents(wikipedia_loader)
character_final_documents = charater_text_splitter.split_documents(wikipedia_loader)'''

#Json data#
json_data = requests.get("https://api.sampleapis.com/coffee/hot").json()
final_documents = json_splitter.split_json(json_data)
print(final_documents)





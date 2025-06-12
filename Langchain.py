#langchain and openAI
#RAG - Retrieval Augmented Generation

###Steps###
'''
1. Loading dataset like image pdf, text Data Ingestion
2. Split into text chunks
3. Embedding - converting text into vectors
4. Store it into Vector DB like chromadb, FAISS, astradb
'''

##Loader##
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader

##Text chunks##
from langchain_text_splitters import RecursiveCharacterTextSplitter


#text loader#
loader = TextLoader('speech.txt')
text_loader = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
final_documents = text_splitter.split_documents(text_loader)



#PDF loader#
loader = PyPDFLoader('basic-text.pdf')
pdf_loader = loader.load()
final_documents = text_splitter.split_documents(pdf_loader)


#Web based loader#
loader = WebBaseLoader(
    web_paths=('https://www.moneycontrol.com',),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_ = ('clearfix MT10')
    ))
)
web_loader = loader.load()
final_documents = text_splitter.split_documents(web_loader)

#Arxiv#
loader = ArxivLoader(
    query="1706.03762",
    load_max_docs=2
)
arxiv_loader = loader.load()
final_documents = text_splitter.split_documents(arxiv_loader)

#WikiPedia#
loader = WikipediaLoader(query="Generative AI", load_max_docs=2)
wikipedia_loader = loader.load()
final_documents = text_splitter.split_documents(wikipedia_loader)




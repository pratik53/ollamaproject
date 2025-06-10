#LLM powered chatbot
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder

load_dotenv()

store = {}

groq_api_key = os.getenv('GROQ_API_KEY')

model = ChatGroq(model="gemma2-9b-it", groq_api_key = groq_api_key)
model.invoke([HumanMessage(content="Hi, My name is Pratik and I am Chief AI Engineer")])

model.invoke([
    HumanMessage(content="Hi, My name is Pratik and I am Chief AI Engineer"),
    AIMessage(content="Hello Pratik! How can I help you"),
    HumanMessage(content="Hey whats my name and what do I do?")
])


def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

config = {"configurable":{"session_id":"chat1"}}

with_message_history = RunnableWithMessageHistory(model, get_session_history)

with_message_history.invoke(
    [HumanMessage(content="Hi, I am Pratik. I am an AI engineer")],
    config=config
)

response =with_message_history.invoke(
    [HumanMessage(content="What is my name")],
    config=config
)



config = {"configurable":{"session_id":"chat2"}}
response =with_message_history.invoke(
    [HumanMessage(content="What is my name")],
    config=config
)

prompt = ChatMessagePromptTemplate([
    ('system', 'You are a helpful assistant. Answer all the questions to the best of your ability.'),
    MessagesPlaceholder(variable_name='message')
])

# Create the chain
chain = prompt | model

# Invoke the chain
response = chain.invoke({"message": [HumanMessage(content="My name is Pratik")]})

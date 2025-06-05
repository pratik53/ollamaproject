import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load env
load_dotenv()

from langchain_groq import ChatGroq

model = ChatGroq(model="gemma2-9b-it", groq_api_key = os.getenv("GROQ_API_KEY"))

messages = [
    SystemMessage(content="Transalate the following from English to Hindi"),
    HumanMessage(content="Hello How are you?")
]

result = model.invoke(messages)
parser = StrOutputParser()
result = parser.invoke(result)

chain = model|parser
result = chain.invoke(messages)


#Prompt template
generic_template = "Transalte the follwing into {language}"
prompt = ChatPromptTemplate.from_messages(
    [('system',generic_template),('user',"{text}")]
)

result = prompt.invoke({'language':"Hindi","text":"Who are you"})

chain = prompt|model|parser
result = chain.invoke({"language":"Marathi","text":"Hello, How are you"})
print(result)


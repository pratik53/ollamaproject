from langchain_community.llms import HuggingFaceEndpoint
import os

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    max_new_tokens=100,
    do_sample=False,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

response = llm.invoke("What is Deep Learning?")
print(response)

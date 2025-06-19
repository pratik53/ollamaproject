# translator_app.py

import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
import os

# --- Set API Token (use environment variable or paste directly) ---
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "your_huggingface_token_here")

# Streamlit UI
st.title("🌐 Translator with HuggingFace Endpoint")

src_text = st.text_area("Enter text to translate:", height=150)
endpoint_url = st.text_input("HuggingFace Endpoint URL", value="https://abc123.us-east-1.aws.endpoints.huggingface.cloud")

if st.button("Translate"):
    if not HUGGINGFACE_API_TOKEN or not endpoint_url:
        st.error("Please provide Hugging Face token and endpoint URL.")
    else:
        try:
            llm = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
            )
            result = llm(src_text)
            st.success("Translation:")
            st.write(result)
        except Exception as e:
            st.error(f"Translation failed: {e}")

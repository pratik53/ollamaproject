# translator_app.py

import streamlit as st
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from huggingface_hub import login

# Optional: Login if using private models or higher rate limits
# login("your_huggingface_token")

# Load translation pipeline using Hugging Face Transformers
@st.cache_resource
def get_translator(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    return pipeline("translation", model=model_name)

# Streamlit UI
st.title("🌍 Language Translator with Hugging Face + LangChain")

src_text = st.text_area("Enter text to translate:", height=150)
src_lang = st.selectbox("From language", ["en", "fr", "de", "es", "hi"])
tgt_lang = st.selectbox("To language", ["hi", "en", "fr", "de", "es"])

if st.button("Translate"):
    try:
        if src_lang == tgt_lang:
            st.warning("Source and target language cannot be the same.")
        else:
            translator = get_translator(src_lang, tgt_lang)
            result = translator(src_text)
            st.success("Translation:")
            st.write(result[0]['translation_text'])
    except Exception as e:
        st.error(f"Translation failed: {e}")

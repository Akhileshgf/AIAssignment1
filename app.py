import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Load Gemini LLM
client = genai.Client(api_key=google_api_key)

# Streamlit default theme
st.title("PRA COREP Own Funds Assistant (Gemini)")

st.write("""
This assistant answers questions only about **Own Funds** reporting.
If you provide Tier1Capital, Tier2Capital, and TotalCapital numbers,
it will show them in a table and run validation checks.
""")

# --- Preload Own Funds PDF into vector DB ---
loader = PyPDFLoader("Own_Funds_09-02-2026.pdf")  # path to your PRA Own Funds PDF
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Use Gemini embeddings
embeddings = client.models.embed_content(model="gemini-embedding-001", contents='PRA Guidelines',config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))

# Store in FAISS vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# --- Helper: parse numbers ---
def parse_numbers(text):
    import re
    nums = re.findall(r"\d+", text)
    if len(nums) >= 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    return None

# Chat UI
user = st.chat_message("User")
bot = st.chat_message("Assistant")

prompt1 = st.chat_input("Ask about Own Funds or provide Tier1, Tier2, Total numbers:")

if prompt1:
    user.write(f"User: {prompt1}")
    parsed = parse_numbers(prompt1)

    if parsed:
        tier1, tier2, total = parsed
        bot.write("Assistant: Own Funds Report")
        st.table({
            "Item": ["Tier1Capital", "Tier2Capital", "TotalCapital"],
            "Amount": [tier1, tier2, total]
        })
        if total == tier1 + tier2:
            bot.write("Validation Passed! ")
        else:
            bot.write("Validation Failed  (Total ≠ Tier1 + Tier2)")
    else:
        if "own funds" in prompt1.lower() or "tier" in prompt1.lower():
            # Retrieve context from vector DB
            docs = retriever.get_relevant_documents(prompt1)
            context = " ".join([d.page_content for d in docs])
            response = client.models.generate_content(model="gemini-2.5-flash", contents=f"Answer only about Own Funds. Context: {context}. Question: {prompt1}") 
            bot.write(f"Assistant: {response.content}")
        else:
            bot.write("Assistant: I only answer questions related to Own Funds reporting.")












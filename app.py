from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))
print("LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title("LangChain Chatbot using Groq")
input_text = st.text_input("Ask anything...")

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Response
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
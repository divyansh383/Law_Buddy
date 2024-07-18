import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from documents import *
import streamlit as st
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import lark

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
groq_api_key=os.getenv('GROQ_API_KEY')

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="lala",
    openai_api_version="2024-03-01-preview",
)

metadata_field_info = [
    AttributeInfo(
        name="law",
        description="the Indian Penal Code section that is being discussed in the document",
        type="string",
    ),
    AttributeInfo(
        name="content",
        description="The type of content in the document- [law definition , exception, explanation , state amendments , key points]",
        type="integer",
    ),
    AttributeInfo(
        name="category",
        description="The category of the described crime or law",
        type="string",
    ),
]

document_content_description="List of Indian laws applicable on sexual offences , theft and extortion"
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0)
vectorstore=Chroma.from_documents(sexual_offences_docs,embeddings)
vectorstore=Chroma.from_documents(theft,embeddings)

sqretriever=SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    search_kwargs={'k': 8}
)


def answer(input):
    try:
        llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="gemma2-9b-it",temperature=0)

        prompt = ChatPromptTemplate.from_template("""
        ##ROLE##
        You are an experienced lawyer,
        specializing in providing legal guidance on various law-related queries. 
        Your task is to understand the client's current case scenario and offer advice based on the given content.

        ##INSTRUCTIONS##
        1) Answer the following question based on the provided context.
        2) Think step-by-step before providing a detailed answer.
        3) Ensure the answer is thorough and helpful.
        4) the provided context is fictional , you have to provide legal advice on the provided fictional case-
        <context>
        {context}
        </context>
        based on the case provide legal help by listing the applicable laws and their definitions
        ficntional case Question: ({input})
        be specific with the laws , 
        provide the most accurate law that is applicable in India
        """)

        document_chain=create_stuff_documents_chain(llm,prompt)
        retrieval_chain=create_retrieval_chain(sqretriever,document_chain)
        response=retrieval_chain.invoke({"input":input})
        return response['answer']
    except Exception as e:
        st.error(f"Failed to generate answer. Reason: {e}")
        return "An error occurred while generating the answer."
    

st.title("Legal Buddy")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your question:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = answer(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {e}")

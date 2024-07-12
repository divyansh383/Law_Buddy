import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import shutil
# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image
# import io

# Load environment variables
load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="lala",
    openai_api_version="2024-03-01-preview",
)


def add_vector_database(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents=text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embeddings)
    database = FAISS.load_local('first__vector',embeddings, allow_dangerous_deserialization= True)
    database.merge_from(db)
    database.save_local('first__vector')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
# gemma-7b-it
# mixtral-8x7b-32768
# gemma2-9b-it
# Llama3-8b-8192
prompt = ChatPromptTemplate.from_template(
"""
##ROLE##
You are an experienced lawyer,
specializing in providing legal guidance on various law-related queries. 
Your task is to understand the client's current case scenario and offer advice based on the given content.

##INSTRUCTIONS##
1) Answer the following question based only on the provided context.
2) Think step-by-step before providing a detailed answer.
3) Ensure the answer is thorough and helpful.
4) the provided context is fictional , you have to provide legal advice on the provided fictional case-
<context>
{context}
</context>
based on the case provide legal help by providing 
1) Laws
2) case studies
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

def delete_vector_database():
    try:
        db = FAISS.load_local('first__vector', embeddings, allow_dangerous_deserialization=True)
        for i in range(db.index.ntotal - 1, -1, -1):
            try:
                doc_id = db.index_to_docstore_id[i]
                db.delete([doc_id])
            except Exception as e:
                st.error(f"Error deleting index {i}: {e}")
        db.save_local('first__vector')
    except Exception as e:
        st.error(f"Failed to delete vector database. Reason: {e}")

def answer(input):
    try:
        database = FAISS.load_local('first__vector', embeddings, allow_dangerous_deserialization=True)
        retriever = database.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": input})
        return response['answer']
    except Exception as e:
        st.error(f"Failed to generate answer. Reason: {e}")
        return "An error occurred while generating the answer."

def clear_uploaded_pdfs(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"Failed to delete {file_path}. Reason: {e}")

# Streamlit app
st.title("Legal Buddy")

if not os.path.exists("uploaded_pdfs"):
    os.makedirs("uploaded_pdfs")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    try:
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded and saved PDF: {uploaded_file.name}")
        add_vector_database("uploaded_pdfs")
        st.success("PDF has been added to the vector database.")
        clear_uploaded_pdfs("uploaded_pdfs")
        st.success("Uploaded files have been cleared from the local directory.")
    except Exception as e:
        st.error(f"Failed to save or process the uploaded PDF. Reason: {e}")

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

if st.button("Delete Vector Database"):
    try:
        delete_vector_database()
        st.success("Vector database has been deleted.")
    except Exception as e:
        st.error(f"Failed to delete vector database. Reason: {e}")

import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
groq_api_key=os.getenv('GROQ_API_KEY')


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="lala",
    openai_api_version="2024-03-01-preview",
)

def add_vector_database(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents=text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embeddings)
    database = FAISS.load_local('first__vector',embeddings, allow_dangerous_deserialization= True)
    database.merge_from(db)
    database.save_local('first__vector')

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

document_chain=create_stuff_documents_chain(llm,prompt)

def delete_vector_database():
    db = FAISS.load_local('first__vector',embeddings, allow_dangerous_deserialization= True)
    for i in range(db.index.ntotal - 1, -1, -1):
        try:
            # Get the docstore ID for the current index
            doc_id = db.index_to_docstore_id[i]
            
            # Delete the entry from FAISS index and docstore
            db.delete([doc_id])
            
            # Add the index to the deleted list
            # deleted_indices.append(i)
        except Exception as e:
            print(f"Error deleting index {i}: {e}")

def answer(input):
    database = FAISS.load_local('first__vector',embeddings, allow_dangerous_deserialization= True)
    retriever=database.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":input})

    return response['answer']

import shutil
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
st.title("Document-Based Question Answering")

# Create the directory if it does not exist
if not os.path.exists("uploaded_pdfs"):
    os.makedirs("uploaded_pdfs")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Save uploaded PDF
if uploaded_file is not None:
    pdf_path = os.path.join("uploaded_pdfs", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded and saved PDF: {uploaded_file.name}")

    # Update the vector database
    add_vector_database("uploaded_pdfs")
    st.success(f"PDF has been added to the vector database.")

    # Clear the uploaded_pdfs directory
    clear_uploaded_pdfs("uploaded_pdfs")
    st.success("Uploaded files have been cleared from the local directory.")

# Text input for question
question = st.text_input("Enter your question:")

# Submit button
if st.button("Get Answer"):
    if question:
        try:
            response = answer(question)
            st.success("Here's the answer:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a question.")

if st.button("Delete Vector Database"):
    try:
        delete_vector_database()
        st.success("Vector database has been deleted.")
    except Exception as e:
        st.error(f"Failed to delete vector database. Reason: {e}")
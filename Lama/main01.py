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

# Function to extract text from images in a PDF
# def extract_text_from_images(pdf_path):
#     try:
#         images = convert_from_path(pdf_path)
#         text = ""
#         for image in images:
#             text += pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         st.error(f"Failed to extract text from images. Reason: {e}")
#         return ""

# # Function to add vector database
# def add_vector_database(directory_path):
#     try:
#         # Load PDFs from directory
#         loader = PyPDFDirectoryLoader(directory_path)
#         docs = loader.load()

#         # Extract text from PDFs if not present
#         for doc in docs:
#             if not hasattr(doc, 'text') or doc.text.strip() == "":
#                 try:
#                     # Correct path handling
#                     pdf_path = os.path.join(directory_path, os.path.basename(doc.metadata['source']))
#                     text = extract_text_from_images(pdf_path)
#                     doc.text = text
#                 except Exception as e:
#                     st.error(f"Failed to extract text for document {doc.metadata.get('source', 'unknown')}. Reason: {e}")
#         # Split documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#         documents = text_splitter.split_documents(docs)

#         # Check if documents are not empty
#         if not documents:
#             st.error("No documents found to process.")
#             return

#         try:
#             # Create FAISS database from documents
#             db = FAISS.from_documents(documents, embeddings)
#             # Load existing database if exists
#             database = FAISS.load_local('first__vector', embeddings, allow_dangerous_deserialization=True)
#             # Merge new documents into the existing database
#             database.merge_from(db)
#             # Save updated database
#             database.save_local('first__vector')
#             st.success("PDF has been added to the vector database.")
#         except IndexError:
#             st.error("Embeddings generation failed. Ensure the documents are correctly loaded and processed.")
#     except Exception as e:
#         st.error(f"Failed to add vector database. Reason: {e}")

def add_vector_database(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents=text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, embeddings)
    database = FAISS.load_local('first__vector',embeddings, allow_dangerous_deserialization= True)
    database.merge_from(db)
    database.save_local('first__vector')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
You are a legal expert and based on the query, 
go through the context 
<context>
{context}
</context>
and respond to the query based by providing applicable laws , legal help and similar case studies
Question: {input}
""")

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

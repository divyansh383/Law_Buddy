import streamlit as st
from docs import load_documents, save_document
from langchain_core.documents import Document

# Define the admin panel function
def admin_panel():
    st.title("Admin Panel")

    with st.expander("Add Data"):
        with st.form("add_data_form"):
            document_content = st.text_area("Document Content", height=200)
            st.text("Metadata:")
            law_name = st.text_input("Law Name", placeholder="Enter the full law name here")
            content_description = st.text_input("Content Description", placeholder="Describe the content of the added text")
            category = st.text_input("Category", placeholder="Enter the category of the law here")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                try:
                    add_document(document_content, law_name, content_description, category)
                    st.success("Data added successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                st.experimental_rerun()

def add_document(page_content, law_name, content_description, category):
    new_doc = Document(
        page_content=page_content,
        metadata={
            "law": law_name,
            "content": content_description + " for the law - " + law_name,
            "category": category
        }
    )
    save_document(new_doc)  

import json
import os
from langchain_core.documents import Document

# Mongo DB setup
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
MONGO_URI=os.getenv('MONGO_URI')
client=MongoClient(MONGO_URI)
db=client['legal-buddy-documents']
collection=db['legal-buddy-collection']

def load_documents():
    """Load documents from MongoDB."""
    documents = collection.find()
    return [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in documents]

def save_documents(documents):
    """Append documents to MongoDB."""
    # Prepare documents for insertion
    docs_to_insert = [{
        'page_content': doc.page_content,
        'metadata': doc.metadata
    } for doc in documents]
    
    # Insert documents
    collection.insert_many(docs_to_insert)

## Sample Usage

if __name__ == "__main__":
    # Print existing documents
    print("Loading documents...")
    documents = load_documents()
    for doc in documents:
        print(f"Page Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)
    
#     # Add a new document
#     print("Adding new document...")
#     new_doc = Document(
#         page_content="Sample page content",
#         metadata={
#             "law": "Sample Law",
#             "content": "Sample content description",
#             "category": "Sample Category"
#         }
#     )
    
#     # Load existing documents, add the new one, and save
#     documents.append(new_doc)
#     save_documents([new_doc])
    
#     # Print documents again to see the updated list
#     print("Loading documents after adding new one...")
#     documents = load_documents()
#     for doc in documents:
#         print(f"Page Content: {doc.page_content}")
#         print(f"Metadata: {doc.metadata}")
#         print("-" * 40)
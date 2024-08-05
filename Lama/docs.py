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

def save_documents(document):
    """Insert a single document to MongoDB."""
    doc_to_insert = {
        'page_content': document.page_content,
        'metadata': document.metadata
    }
    
    # Insert the document
    collection.insert_one(doc_to_insert)


## Sample Usage

if __name__ == "__main__":
    # Print existing documents
    # print("Loading documents...")
    # documents = load_documents()
    # for doc in documents:
    #     print(f"Page Content: {doc.page_content}")
    #     print(f"Metadata: {doc.metadata}")
    #     print("-" * 40)
    

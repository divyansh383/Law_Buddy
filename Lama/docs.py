import json
import os
from langchain_core.documents import Document

# Path to the JSON file
DOCUMENTS_FILE = 'documents.json'

def load_documents():
    """Load documents from a JSON file."""
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, 'r') as file:
            data = json.load(file)
            return [Document(page_content=doc['page_content'], metadata=doc['metadata']) for doc in data]
    return []

def save_documents(documents):
    """Save documents to a JSON file."""
    data = [{
        'page_content': doc.page_content,
        'metadata': doc.metadata
    } for doc in documents]
    with open(DOCUMENTS_FILE, 'w') as file:
        json.dump(data, file, indent=4)


documents = load_documents()

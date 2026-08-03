import fitz
import glob
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS

import providers

text_embeddings = providers.get_embeddings()


semantic_chunker = SemanticChunker(text_embeddings, breakpoint_threshold_type="percentile")

def extract_text(pdf_file):
    doc = fitz.open(pdf_file)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # "text" gives plain text
        all_text += text + "\n"
    doc.close()
    return all_text

def create_document(pdf_files):
    # Initialize text splitter (commented out - using semantic_chunker instead)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len,
    #     separators=["\n\n", "\n", " ", ""]
    # )

    all_documents = []

    # Loop through each PDF
    for pdf in pdf_files:
        print(f"Processing: {pdf}")
        try:
            # Read text from PDF
            full_text = extract_text(pdf)
            print(f"PDF file: {len(full_text)} characters")

            # Use semantic_chunker to split into semantic chunks
            # create_documents accepts a list of strings (text content)
            documents = semantic_chunker.create_documents([full_text])
            
            # Add metadata to each document chunk
            for doc in documents:
                doc.metadata = {"source": os.path.basename(pdf)}  # filename.pdf

            # Old approach using RecursiveCharacterTextSplitter (commented out)
            # chunks = text_splitter.split_text(full_text)
            # documents = [
            #     Document(
            #         page_content=chunk,
            #         metadata={"source": os.path.basename(pdf)}  # filename.pdf
            #     )
            #     for chunk in chunks
            # ]

            all_documents.extend(documents)

        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    return all_documents

# Open the PDFs, relative to this file so the script runs on any machine
pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf", "*.pdf")
pdf_files = glob.glob(pdf_path)
if not pdf_files:
    raise SystemExit(f"No PDFs found at {pdf_path}")
all_documents = create_document(pdf_files)
print(f"Total documents created: {len(all_documents)}")

# Create FAISS vector store from documents
print("Creating embeddings and building FAISS index...")
import time
start_time = time.time()
vector_store = FAISS.from_documents(all_documents, text_embeddings)
elapsed_time = time.time() - start_time
print(f"Embeddings created and FAISS index built in {elapsed_time:.2f} seconds")

print("Saving vector store to disk...")
vector_store.save_local("faiss_index")
print("✓ FAISS vector store created and saved to 'faiss_index' directory")
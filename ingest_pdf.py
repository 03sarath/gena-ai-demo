import sys
import os
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Usage: python ingest_pdf.py path/to/leave_policy.pdf

chroma_client = chromadb.Client(Settings(
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
))
collection_name = os.getenv("CHROMA_COLLECTION_NAME", "leave_policy_pdfs")
embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5"))

def extract_pdf_chunks(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # Split text into chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            meta = {
                "page": page_num + 1,
                "source": os.path.basename(pdf_path)
            }
            chunks.append((chunk, meta))
    return chunks

def main(pdf_path):
    collection = chroma_client.get_or_create_collection(collection_name)
    chunks = extract_pdf_chunks(pdf_path)
    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    print(f"Extracted {len(texts)} chunks from {pdf_path}")
    # Embed and add to ChromaDB
    embeddings = list(embedder.embed(texts))
    ids = [f"{os.path.basename(pdf_path)}_p{m['page']}_c{i}" for i, m in enumerate(metadatas)]
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Ingested {len(texts)} chunks into ChromaDB.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_pdf.py path/to/leave_policy.pdf")
        sys.exit(1)
    main(sys.argv[1]) 
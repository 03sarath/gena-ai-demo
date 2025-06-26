import sys
import os
import chromadb
from fastembed import TextEmbedding
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_pdf_chunks(pdf_path, chunk_size=500):
    """Extract text chunks from PDF"""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        total_pages = len(reader.pages)
        print(f"📄 Processing {total_pages} pages...")
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or text.strip() == "":
                continue
                
            # Split text into chunks
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size].strip()
                if chunk:
                    meta = {
                        "page": page_num + 1,
                        "source": os.path.basename(pdf_path),
                        "chunk_id": i // chunk_size
                    }
                    chunks.append((chunk, meta))
        
        print(f"✅ Extracted {len(chunks)} text chunks")
        return chunks
        
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return []

def main(pdf_path):
    """Main ingestion function"""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Setup ChromaDB with absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_db_path = os.path.join(current_dir, "chroma_db")
    collection_name = "leave_policy_pdfs"
    
    print(f"🔧 ChromaDB Settings:")
    print(f"  Path: {chroma_db_path}")
    print(f"  Collection: {collection_name}")
    
    try:
        # Create ChromaDB client
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Delete existing collection if it exists
        try:
            existing_collection = chroma_client.get_collection(collection_name)
            print(f"🗑️  Deleting existing collection: {collection_name}")
            chroma_client.delete_collection(collection_name)
        except:
            print(f"📝 Creating new collection: {collection_name}")
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Leave policy documents"}
        )
        
        # Extract chunks from PDF
        chunks = extract_pdf_chunks(pdf_path)
        if not chunks:
            print("❌ No chunks extracted from PDF")
            sys.exit(1)
        
        # Prepare data
        texts = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]
        
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        
        # Initialize FastEmbed
        embedder = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        
        # Generate embeddings
        embeddings = list(embedder.embed(texts))
        
        # Create unique IDs
        ids = [f"{os.path.basename(pdf_path)}_p{m['page']}_c{m['chunk_id']}" for m in metadatas]
        
        print(f"💾 Adding {len(texts)} documents to ChromaDB...")
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Verify the data was added
        count = collection.count()
        print(f"✅ Successfully ingested {count} documents into ChromaDB")
        
        # Test query
        print(f"🧪 Testing query...")
        test_embedding = list(embedder.embed(["annual leave"]))[0]
        test_results = collection.query(
            query_embeddings=[test_embedding],
            n_results=1,
            include=["metadatas", "documents"]
        )
        
        if test_results["documents"][0]:
            print(f"✅ Query test successful - found {len(test_results['documents'][0])} results")
            print(f"📄 Sample document: {test_results['documents'][0][0][:100]}...")
        else:
            print("⚠️  Query test returned no results")
        
        # Check if files were created
        if os.path.exists(chroma_db_path):
            files = os.listdir(chroma_db_path)
            print(f"📁 ChromaDB files created: {len(files)} files")
            for file in files:
                print(f"  - {file}")
        else:
            print("⚠️  ChromaDB directory not found after ingestion")
            
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_new.py path/to/leave_policy.pdf")
        sys.exit(1)
    main(sys.argv[1]) 
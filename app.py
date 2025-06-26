from flask import Flask, request, jsonify
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize ChromaDB client with absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_path = os.path.join(current_dir, "chroma_db")
print(f"ChromaDB path: {chroma_db_path}")

chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection_name = "leave_policy_pdfs"

# Initialize FastEmbed model
embedder = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Custom API Configuration
CUSTOM_API_ENDPOINT = os.getenv("CUSTOM_API_ENDPOINT")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")

def embed_text(text):
    """Embed text using FastEmbed"""
    return list(embedder.embed([text]))[0]

def query_chroma_db(query_embedding, k=5):
    """Query ChromaDB for similar documents"""
    try:
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Leave policy documents"}
        )
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )
        
        print(f"Query returned {len(results['documents'][0])} documents")
        return results
        
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

def generate_llm_response(user_query, top_results):
    """Generate response using custom API"""
    # Check if we have documents
    if not top_results["documents"][0] or len(top_results["documents"][0]) == 0:
        return "I don't have any leave policy documents in my database to answer your question. Please make sure to ingest your leave policy PDF first."
    
    # Combine document chunks
    context_parts = []
    for doc, meta in zip(top_results["documents"][0], top_results["metadatas"][0]):
        context_parts.append(f"Document: {doc}\nMetadata: {meta}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""
You are an assistant for answering questions about the company's leave policy. Use the following context from the policy documents to answer the user's question in a clear and concise way.

Context:
{context}

User question: {user_query}

Answer:"""
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CUSTOM_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            CUSTOM_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return f"Error: API returned status code {response.status_code}"
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

@app.route("/ask", methods=["POST"])
def ask():
    """Main API endpoint"""
    try:
        data = request.json
        user_question = data.get("question")
        
        if not user_question:
            return jsonify({"error": "Missing 'question' in request."}), 400
        
        print(f"Processing question: {user_question}")
        
        # Embed the question
        query_embedding = embed_text(user_question)
        print("Question embedded successfully")
        
        # Query ChromaDB
        top_results = query_chroma_db(query_embedding, k=5)
        print(f"Retrieved {len(top_results['documents'][0])} documents from ChromaDB")
        
        # Generate response
        answer = generate_llm_response(user_question, top_results)
        
        return jsonify({
            "answer": answer, 
            "top_results": top_results,
            # "debug_info": {
            #     "chroma_db_path": chroma_db_path,
            #     "collection_name": collection_name,
            #     "documents_found": len(top_results["documents"][0])
            # }
        })
        
    except Exception as e:
        print(f"Error in /ask endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        collection = chroma_client.get_or_create_collection(collection_name)
        count = collection.count()
        return jsonify({
            "status": "healthy",
            "chroma_db_path": chroma_db_path,
            "collection_name": collection_name,
            "document_count": count
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting Flask API...")
    print(f"ChromaDB path: {chroma_db_path}")
    
    # Test ChromaDB connection
    try:
        collection = chroma_client.get_or_create_collection(collection_name)
        count = collection.count()
        print(f"ChromaDB connected. Collection '{collection_name}' has {count} documents.")
    except Exception as e:
        print(f"Warning: ChromaDB connection issue: {e}")
    
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug) 
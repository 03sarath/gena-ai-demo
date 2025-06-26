from flask import Flask, request, jsonify
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize ChromaDB client with environment variables
chroma_client = chromadb.Client(Settings(
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
))
collection_name = os.getenv("CHROMA_COLLECTION_NAME", "leave_policy_pdfs")

# Initialize FastEmbed model with environment variable
embedder = TextEmbedding(model_name=os.getenv("FASTEMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5"))

# Custom API Configuration
CUSTOM_API_ENDPOINT = os.getenv("CUSTOM_API_ENDPOINT", "https://api.generate.engine.psitrontech.com/v2/llm/invoke")
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "your_api_key_here")

# Helper: Embed a text string
def embed_text(text):
    return list(embedder.embed([text]))[0]

# Helper: Query ChromaDB for top-k similar documents
def query_chroma_db(query_embedding, k=5):
    collection = chroma_client.get_or_create_collection(collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )
    return results

# Helper: Use Custom API to generate a meaningful answer
def generate_llm_response(user_query, top_results):
    context = "\n".join([
        f"Document: {doc}\nMetadata: {meta}" for doc, meta in zip(top_results["documents"][0], top_results["metadatas"][0])
    ])
    prompt = f"""
You are an assistant for answering questions about the company's leave policy. Use the following context from the policy documents to answer the user's question in a clear and concise way.\n\nContext:\n{context}\n\nUser question: {user_query}\n\nAnswer:"""
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CUSTOM_API_KEY}"
        }
        
        payload = {
            "model": os.getenv("MODEL_NAME", "gpt-4"),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": int(os.getenv("MAX_TOKENS", "500")),
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
            
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return f"Error making API request: {str(e)}"
    except Exception as e:
        print(f"General Error: {e}")
        return f"Error generating response: {str(e)}"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question")
    if not user_question:
        return jsonify({"error": "Missing 'question' in request."}), 400
    # Embed the user question
    query_embedding = embed_text(user_question)
    # Query ChromaDB for top 5 results
    top_results = query_chroma_db(query_embedding, k=5)
    # Generate LLM response
    answer = generate_llm_response(user_question, top_results)
    return jsonify({"answer": answer, "top_results": top_results})

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(host=host, port=port, debug=debug) 
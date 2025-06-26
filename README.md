# Leave Policy QA API

This project provides a Flask API for answering questions about your company's leave policy using PDF documents. It uses FastEmbed for embedding, ChromaDB as a vector database, and Psitron Tech's LLM API for generating meaningful answers.

## Features
- Upload and embed leave policy PDFs (metadata stored in ChromaDB)
- Query API with natural language questions
- Returns top 5 relevant policy sections and a concise answer from Psitron Tech LLM
- Environment-based configuration using .env file
- Enhanced reasoning capabilities with custom LLM endpoint

## Quick Start (Automated Setup)

### Option 1: Automated Setup Script
```bash
# Run the automated setup script
python setup_venv.py
```

This script will:
- ✅ Check Python version compatibility
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Create .env file from template
- ✅ Provide next steps

### Option 2: Manual Setup

#### 1. **Create and activate virtual environment:**

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### 2. **Install dependencies:**
```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

#### 3. **Set up environment variables:**
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit the .env file with your settings
# (Use any text editor to open .env)
```

## Environment Configuration

Edit your `.env` file with the following settings:

```bash
# Custom API Configuration (Psitron Tech)
CUSTOM_API_ENDPOINT=https://api.generate.engine.psitrontech.com/v2/llm/invoke
CUSTOM_API_KEY= your-api-key

# Optional: Customize other settings
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
CHROMA_PERSIST_DIRECTORY=chroma_db
CHROMA_COLLECTION_NAME=leave_policy_pdfs
FASTEMBED_MODEL_NAME=BAAI/bge-base-en-v1.5
MODEL_NAME=gpt-4
MAX_TOKENS=500
```

## Custom LLM API Benefits

This API uses **Psitron Tech's LLM endpoint** for enhanced capabilities:

- **🎯 Custom Infrastructure**: Dedicated API endpoint for better performance
- **📚 Reliable Service**: Stable and consistent API responses
- **🔍 Enhanced Analysis**: More detailed and nuanced answers
- **💡 Cost Effective**: Optimized pricing for enterprise use
- **📝 High Quality Output**: Professional and well-structured responses

## Usage

### 1. **Activate virtual environment (if not already active):**

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 2. **Ingest your PDF:**
```bash
python ingest_pdf.py leave_policy.pdf
```

### 3. **Start the API server:**
```bash
python app.py
```

### 4. **Ask a question:**
Send a POST request to `/ask` with a JSON body:
```json
{
  "question": "How many days of annual leave am I entitled to?"
}
```

Example using `curl`:
```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What is the maternity leave policy?"}'
```

## Testing with Postman

### Setting Up Postman

1. **Download and Install Postman** from [postman.com](https://www.postman.com/downloads/)

2. **Create a New Collection:**
   - Click "New" → "Collection"
   - Name it "Leave Policy QA API"

3. **Create a New Request:**
   - Click "Add request" in your collection
   - Name it "Ask Question"

### API Request Configuration

#### **Request Details:**
- **Method:** `POST`
- **URL:** `http://127.0.0.1:5000/ask`
- **Headers:**
  ```
  Content-Type: application/json
  ```

#### **Request Body (JSON):**
```json
{
  "question": "How many days of annual leave am I entitled to?"
}
```

### Example Requests

#### **1. Annual Leave Query:**
```json
{
  "question": "What is the annual leave entitlement for employees?"
}
```

#### **2. Maternity Leave Query:**
```json
{
  "question": "What are the maternity leave benefits and duration?"
}
```

#### **3. Sick Leave Query:**
```json
{
  "question": "How many sick days can I take per year?"
}
```

#### **4. Leave Application Process:**
```json
{
  "question": "What is the process for applying for leave?"
}
```

#### **5. Complex Policy Questions:**
```json
{
  "question": "What happens if I need to take emergency leave during a holiday period?"
}
```

### Expected Response Format

#### **Successful Response (200 OK):**
```json
{
  "answer": "Based on the leave policy, employees are entitled to 25 days of annual leave per year. This includes public holidays and can be taken in accordance with the company's leave application process. The policy also outlines specific procedures for requesting leave and approval workflows.",
  "top_results": {
    "documents": [
      ["Document chunk 1 content..."],
      ["Document chunk 2 content..."],
      ["Document chunk 3 content..."],
      ["Document chunk 4 content..."],
      ["Document chunk 5 content..."]
    ],
    "metadatas": [
      [{"page": 1, "source": "leave_policy.pdf"}],
      [{"page": 2, "source": "leave_policy.pdf"}],
      [{"page": 3, "source": "leave_policy.pdf"}],
      [{"page": 4, "source": "leave_policy.pdf"}],
      [{"page": 5, "source": "leave_policy.pdf"}]
    ],
    "distances": [
      [0.123, 0.234, 0.345, 0.456, 0.567]
    ]
  }
}
```

#### **Error Response (400 Bad Request):**
```json
{
  "error": "Missing 'question' in request."
}
```

### Postman Environment Variables (Optional)

For better organization, you can set up environment variables in Postman:

1. **Create Environment:**
   - Click "Environments" → "New"
   - Name it "Leave Policy API"

2. **Add Variables:**
   - `base_url`: `http://127.0.0.1:5000`
   - `api_endpoint`: `/ask`

3. **Use in Request:**
   - URL: `{{base_url}}{{api_endpoint}}`

### Testing Workflow

1. **Start the Flask API:**
   ```bash
   python app.py
   ```

2. **In Postman:**
   - Set up the request as described above
   - Click "Send"
   - Verify the response format and content

3. **Test Different Questions:**
   - Try various leave-related questions
   - Check response quality and relevance
   - Verify that top_results contain relevant document chunks
   - Test complex scenarios that benefit from the custom LLM

### Troubleshooting Postman Issues

#### **Connection Refused:**
- Ensure Flask API is running (`python app.py`)
- Check if the port (5000) is correct
- Verify the URL is `http://127.0.0.1:5000/ask`

#### **400 Bad Request:**
- Ensure request body is valid JSON
- Check that "question" field is present
- Verify Content-Type header is set to `application/json`

#### **Empty Response:**
- Make sure you've ingested PDF data first
- Check if ChromaDB has data: `python ingest_pdf.py leave_policy.pdf`

#### **Custom API Errors:**
- Verify your API key is correct in the `.env` file
- Check if the custom API endpoint is accessible
- Ensure you have sufficient API credits

## Virtual Environment Management

### Activating the Environment
- **Windows:** `venv\Scripts\activate`
- **Linux/macOS:** `source venv/bin/activate`

### Deactivating the Environment
```bash
deactivate
```

### Installing Additional Packages
```bash
# Make sure virtual environment is activated
pip install package_name
```

### Updating Dependencies
```bash
# Update requirements.txt with new packages
pip freeze > requirements.txt
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUSTOM_API_ENDPOINT` | Psitron Tech API endpoint | https://api.generate.engine.psitrontech.com/v2/llm/invoke |
| `CUSTOM_API_KEY` | Your Psitron Tech API key | Required |
| `FLASK_HOST` | Flask server host | 127.0.0.1 |
| `FLASK_PORT` | Flask server port | 5000 |
| `FLASK_DEBUG` | Enable Flask debug mode | True |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage directory | chroma_db |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name | leave_policy_pdfs |
| `FASTEMBED_MODEL_NAME` | FastEmbed model to use | BAAI/bge-base-en-v1.5 |
| `MODEL_NAME` | LLM model for responses | gpt-4 |
| `MAX_TOKENS` | Max tokens for LLM response | 500 |

## File Structure
- `app.py` - Main Flask API
- `ingest_pdf.py` - PDF ingestion script
- `setup_venv.py` - Automated virtual environment setup script
- `requirements.txt` - Python dependencies
- `env_example.txt` - Example environment variables
- `README.md` - This file

## Troubleshooting

### Virtual Environment Issues
- **"python not found"**: Make sure Python is installed and in your PATH
- **"venv not found"**: Run `python -m venv venv` to create the virtual environment
- **Activation fails**: Use the correct activation command for your OS

### Dependency Issues
- **Installation fails**: Try upgrading pip first: `pip install --upgrade pip`
- **Version conflicts**: Delete `venv` folder and recreate the virtual environment

### API Issues
- **Custom API errors**: Check your API key in the `.env` file
- **ChromaDB errors**: Make sure you've ingested PDF data first

### Postman Issues
- **Connection refused**: Ensure Flask API is running and port is correct
- **Invalid JSON**: Check request body format and Content-Type header
- **Empty responses**: Verify PDF data has been ingested into ChromaDB

### Custom API Specific Issues
- **"API key invalid"**: Verify your Psitron Tech API key is correct
- **"Endpoint not found"**: Check if the API endpoint URL is accessible
- **Rate limiting**: Custom APIs may have different rate limits
- **Network issues**: Ensure your network can access the custom endpoint

## Notes
- Always activate the virtual environment before running the application
- Make sure to ingest your PDF data into ChromaDB before querying
- The API uses Psitron Tech's custom LLM endpoint for enhanced performance
- Custom API provides better reliability and cost-effectiveness
- All configuration is managed through environment variables in the `.env` file
- Postman is recommended for testing and debugging API requests
- The custom endpoint may have different latency characteristics

## License
MIT 
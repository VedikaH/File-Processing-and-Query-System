# File Processing and Query System

## Overview

The File Processing and Query System is a RAG (Retrieval-Augmented Generation) based document intelligence platform that allows users to upload PDF or text documents and perform intelligent queries on their content.

## Tech Stack

- **Backend**: FastAPI
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **LLM**: Groq (Gemma2-9b-it)
- **Deployment**: Uvicorn

## Prerequisites

- Python 3.8+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VedikaH/File-Processing-and-Query-System.git
cd document-query-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_groq_api_key
```

## Running the Application

```bash
python main.py
```

The application will start on `http://localhost:9000`

## How It Works

1. **File Upload**: 
   - Supports PDF and text files
   - Extracts text and table content
   - Generates page-wise embeddings

2. **Semantic Search**:
   - Uses SentenceTransformer for embedding
   - FAISS for efficient vector search
   - Retrieves most relevant document passages

3. **Query Response**:
   - Feeds retrieved context to Groq LLM
   - Generates contextual, accurate responses


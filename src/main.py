import os
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from file_upload import PDFProcessor
from rag_sys import RAGQueryHandler
from models import (
    FileUploadResponse, 
    QueryRequest, 
    QueryResponse
)
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Document Query System", 
    description="RAG-based document intelligence platform"
)

# CORS and middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize processors
pdf_processor = PDFProcessor()
rag_query_handler = RAGQueryHandler()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Serve the main frontend page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or text file
    """
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    file_path = pdf_processor.save_uploaded_file(file)
    
    # For PDFs, count total pages
    total_pages = 0
    if file.filename.lower().endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
    
    return FileUploadResponse(
        filename=file.filename,
        file_path=file_path,
        total_pages=total_pages,
        extraction_status="success"
    )

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a previously uploaded document
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Extract content
    extracted_content = PDFProcessor.extract_content(request.file_path)
    
    # Query using RAG with FAISS
    query_response = rag_query_handler.query_document(
        extracted_content, 
        request.query
    )
    
    return query_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class FileUploadResponse(BaseModel):
    filename: str
    file_path: str
    total_pages: int
    extraction_status: str

class QueryRequest(BaseModel):
    file_path: str
    query: str

class QueryResponse(BaseModel):
    query: str
    response: Dict[str, Any]
    source_pages: List[int]

class ExtractionResult(BaseModel):
    text: List[str]
    tables: List[Dict[str, Any]]

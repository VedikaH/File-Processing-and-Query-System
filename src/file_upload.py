
import os
import pdfplumber
import json
from typing import List, Dict, Any

class PDFProcessor:
    @staticmethod
    def extract_content(file_path: str) -> Dict[str, Any]:
        extraction_result = {
            "text": [],
            "tables": []
        }
        
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        extraction_result["text"].append({
                            "page_number": page_num,
                            "content": page_text
                        })
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            extraction_result["tables"].append({
                                "page_number": page_num,
                                "data": table
                            })
            
            return extraction_result
        
        except Exception as e:
            print(f"Error extracting PDF content: {e}")
            return extraction_result

    @staticmethod
    def save_uploaded_file(file, upload_dir: str = "data"):
        """
        Save uploaded file to specified directory
        
        Args:
            file: Uploaded file object
            upload_dir (str): Directory to save files
        
        Returns:
            str: Full path of saved file
        """
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        return file_path

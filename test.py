import pdfplumber

# Open the PDF
file_path='data\LLM Data Set.pdf'
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
    print(extraction_result)
    
except Exception as e:
    print(f"Error extracting PDF content: {e}")
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from llama_parse import LlamaParse
from openai import OpenAI
import sqlite3
from datetime import datetime
import base64
import json
from pathlib import Path
import io
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY

LLAMA_CLOUD_API_KEY=os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY']=LLAMA_CLOUD_API_KEY

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Processor API")

openai_client = OpenAI(api_key="OPENAI_API_KEY")

parser = LlamaParse(
    api_key='LLAMA_CLOUD_API_KEY',
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en"
)


# Database initialization with category fields
def init_db():
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        category TEXT NOT NULL,
        subcategory TEXT NOT NULL,
        important_info TEXT,
        summary TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        all_extracted TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Models
class DocumentResponse(BaseModel):
    id: int
    filename: str
    category: str
    subcategory: str
    important_info: Optional[str] = None
    summary: Optional[str] = None

def classify_document(text: str, categories: dict) -> tuple[str, str]:
    """Classify document into category and subcategory"""
    logger.info("Starting document classification")
    
    categories_prompt = json.dumps(categories, indent=2)
    prompt = f"""Given the following categories and their subcategories:
    {categories_prompt}
    
    And the following document text:
    {text[:1500]}...
    
    Please determine the main category and subcategory this document belongs to.
    The main category is the top-level category (e.g., IdentityDocuments, FinancialDocuments, etc.).
    Return the response in the following format exactly:
    Main Category: [main category name]
    Subcategory: [specific subcategory name]"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a document classification expert. Respond only with the Main Category and Subcategory in the exact format specified."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        response_text = response.choices[0].message.content.strip()
        lines = response_text.split('\n')
        category = lines[0].split('Main Category: ')[1].strip()
        subcategory = lines[1].split('Subcategory: ')[1].strip()
        
        logger.info(f"Document classified as {category}/{subcategory}")
        return category, subcategory
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error classifying document: {str(e)}")

def load_categories():
    """Load categories from JSON file"""
    try:
        with open('categories.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load category definitions")

def extract_text_from_pdf(file_content: bytes, filename: str) -> Optional[str]:
    """Extract text from PDF using multiple methods"""
    logger.info(f"Starting text extraction for file: {filename}")
    
    # First attempt: LlamaParse
    try:
        logger.debug("Attempting LlamaParse extraction")
        extra_info = {"file_name": filename}
        documents = parser.load_data(file_content, extra_info=extra_info)
        
        if documents and len(documents) > 0:
            extracted_text = documents[0].text
            if extracted_text and len(extracted_text.strip()) > 0:
                logger.info("LlamaParse extraction successful")
                return extracted_text
            else:
                logger.warning("LlamaParse returned empty text")
        else:
            logger.warning("No documents returned from LlamaParse")
            
    except Exception as e:
        logger.error(f"LlamaParse extraction failed: {str(e)}")
    
    # Fallback: PyPDF2
    logger.debug("Attempting PyPDF2 extraction")
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        if text.strip():
            logger.info("PyPDF2 extraction successful")
            return text
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {str(e)}")
    
    raise HTTPException(
        status_code=400, 
        detail="Could not extract text from document using any available method"
    )

def extract_important_info(text: str) -> str:
    """Extract important information from document text"""
    logger.info("Starting important info extraction")
    prompt = """From the following document text, extract only the most important identifying information.
    Focus only on:
    1. Personal identifiers (name, ID numbers if present)
    2. Contact information (email, phone if present)
    3. Key events (dates, locations, causes, etc.)
    4. Company information if any
    5. Bank name information if any
    6. Document-specific key dates
    7. Document reference numbers if any

    Return only the found information in a simple list format with labels. If a piece of information is not found, skip it."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a document information extraction expert. Extract only the key identifying information requested."},
                {"role": "user", "content": f"{prompt}\n\nDocument text:\n{text[:1500]}..."}
            ],
            temperature=0
        )
        logger.info("Successfully extracted important info")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in important info extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting important info: {str(e)}")


def get_cached_summary(doc_id: int) -> Optional[str]:
    """Get cached summary from database if it exists"""
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('SELECT summary FROM documents WHERE id = ?', (doc_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result and result[0] else None

def save_summary_to_db(doc_id: int, summary: str):
    """Save summary to database"""
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('UPDATE documents SET summary = ? WHERE id = ?', (summary, doc_id))
    conn.commit()
    conn.close()

def summarize_document(doc_id: int) -> Optional[str]:
    """Summarize document content using GPT-4"""
    # Check cache first
    cached_summary = get_cached_summary(doc_id)
    if cached_summary:
        logger.info("Retrieved cached summary")
        return cached_summary
    
    # Get document text from database
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('SELECT all_extracted FROM documents WHERE id = ?', (doc_id,))
    result = c.fetchone()
    conn.close()
    
    if not result or not result[0]:
        logger.error(f"No document found with id {doc_id}")
        return None
        
    text = result[0]
    
    prompt = f"""Please provide a comprehensive summary of the following document.
    Focus on the main points, key findings, and important details.
    Keep the summary clear and concise while capturing the essential information.
    Provide in Structured Way with headings and bullet points wherever possible.
    
    Document text:
    {text[:1500]}...
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert document summarizer. Create clear, concise, yet comprehensive summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        save_summary_to_db(doc_id, summary)
        logger.info(f"Generated and saved summary for document {doc_id}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return None

# API Endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a new document"""
    logger.info(f"Received file upload request: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        init_db()
        file_content = await file.read()
        file_size = len(file_content)
        logger.debug(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_content, file.filename)
        if not extracted_text:
            logger.error("Text extraction failed")
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        logger.debug(f"Extracted text length: {len(extracted_text)} characters")
        
        # Load categories and classify document
        categories = load_categories()
        category, subcategory = classify_document(extracted_text, categories)
        
        # Extract important information
        important_info = extract_important_info(extracted_text)
        
        # Save to database
        conn = sqlite3.connect('documents.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO documents (filename, category, subcategory, important_info, all_extracted)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, category, subcategory, important_info, extracted_text))
        doc_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully processed document. ID: {doc_id}")
        
        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            category=category,
            subcategory=subcategory,
            important_info=important_info
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}/important-info", response_model=DocumentResponse)
async def get_important_info(doc_id: int):
    """Get important information for a specific document"""
    conn = sqlite3.connect('documents.db')
    c = conn.cursor()
    c.execute('SELECT filename, category, subcategory, important_info FROM documents WHERE id = ?', (doc_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return DocumentResponse(
        id=doc_id,
        filename=result[0],
        category=result[1],
        subcategory=result[2],
        important_info=result[3]
    )


@app.get("/documents/{doc_id}/summary")
async def get_document_summary(doc_id: int):
    """Get a summary of the document content"""
    summary = summarize_document(doc_id)
    
    if summary is None:
        raise HTTPException(status_code=404, detail="Could not generate summary or document not found")
        
    return {"doc_id": doc_id, "summary": summary}


@app.get("/categories")
async def get_categories():
    """Get all available categories"""
    try:
        categories = load_categories()
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
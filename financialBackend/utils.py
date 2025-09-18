# utils.py
import boto3
import time
import os
import pandas as pd
import PyPDF2
import io

# Initialize the boto3 client lazily
def get_textract_client():
    """Get or create a Textract client with proper region and credentials configuration."""
    try:
        region = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Check if AWS credentials are provided in environment variables
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key_id and aws_secret_access_key:
            # Use explicit credentials from .env file
            return boto3.client(
                "textract",
                region_name=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Use default credential chain (IAM roles, AWS CLI, etc.)
            return boto3.client("textract", region_name=region)
            
    except Exception as e:
        raise Exception(f"Failed to initialize AWS Textract client: {e}")

def extract_text_with_aws_textract_sync(file) -> str:
    """
    Extract text using AWS Textract synchronous API with enhanced accuracy for financial documents.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        Extracted text as string
    """
    try:
        print(f"🔒 Using AWS Textract for secure extraction: {file.name}")
        
        # Read file content
        file_content = file.read()
        file.seek(0)  # Reset file pointer for potential future use
        
        # Initialize Textract client
        textract_client = get_textract_client()
        
        # Use synchronous detect_document_text API with enhanced features
        response = textract_client.detect_document_text(
            Document={'Bytes': file_content}
        )
        
        # Enhanced text extraction with better formatting
        extracted_text = extract_text_with_formatting(response)
        
        # Clean and validate the extracted text
        cleaned_text = clean_financial_text(extracted_text)
        
        return cleaned_text
        
    except Exception as e:
        raise Exception(f"AWS Textract synchronous extraction failed: {e}")

def extract_text_with_formatting(response) -> str:
    """
    Extract text from Textract response with proper formatting for financial documents.
    """
    # Group blocks by page and maintain spatial relationships
    pages = {}
    for block in response.get('Blocks', []):
        if block.get('BlockType') == 'PAGE':
            page_num = block.get('Page', 1)
            pages[page_num] = []
        elif block.get('BlockType') == 'LINE':
            page_num = block.get('Page', 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(block)
    
    # Sort lines by position (top to bottom, left to right)
    extracted_text = ""
    for page_num in sorted(pages.keys()):
        lines = pages[page_num]
        # Sort by Y coordinate (top to bottom), then X coordinate (left to right)
        lines.sort(key=lambda x: (x.get('Geometry', {}).get('BoundingBox', {}).get('Top', 0), 
                                 x.get('Geometry', {}).get('BoundingBox', {}).get('Left', 0)))
        
        for line in lines:
            text = line.get('Text', '').strip()
            if text:
                extracted_text += text + '\n'
        
        # Add page separator for multi-page documents
        if page_num < max(pages.keys()):
            extracted_text += '\n--- PAGE BREAK ---\n'
    
    return extracted_text.strip()

def clean_financial_text(text: str) -> str:
    """
    Clean and enhance text for better financial document processing.
    """
    if not text:
        return text
    
    # Remove excessive whitespace but preserve structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean the line
        cleaned_line = line.strip()
        
        # Skip empty lines but preserve structure
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines and cleaned_lines[-1] != '':
            # Add single empty line for structure
            cleaned_lines.append('')
    
    # Join lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Fix common OCR issues in financial documents
    cleaned_text = fix_common_ocr_errors(cleaned_text)
    
    return cleaned_text

def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors in financial documents.
    """
    # Common OCR replacements for financial documents
    replacements = {
        # Currency symbols
        '€': '€',
        '$': '$',
        '£': '£',
        
        # Numbers and amounts
        'O': '0',  # Letter O to number 0
        'l': '1',  # Lowercase l to number 1
        'I': '1',  # Uppercase I to number 1
        'S': '5',  # Letter S to number 5 (in some fonts)
        
        # Common financial terms
        'lnvoice': 'Invoice',
        'lnv': 'Inv',
        'TotaI': 'Total',
        'Amount': 'Amount',
        'VAT': 'VAT',
        'BTW': 'BTW',
        
        # Date formats
        '0l': '01',
        '02': '02',
        '03': '03',
        '04': '04',
        '05': '05',
        '06': '06',
        '07': '07',
        '08': '08',
        '09': '09',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def extract_text_from_pdf(file) -> str:
    """
    Extract text from PDF file using PyPDF2 with enhanced accuracy.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        Extracted text as string
    """
    try:
        print(f"📄 Using PyPDF2 for local extraction: {file.name}")
        
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text.strip():
                text += f"--- PAGE {page_num + 1} ---\n"
                text += page_text + "\n\n"
        
        # Clean and enhance the extracted text
        cleaned_text = clean_financial_text(text.strip())
        
        return cleaned_text
        
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")

def extract_text_from_file(file) -> str:
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        Extracted text as string
    """
    file_name = file.name.lower()
    
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        raise Exception(f"Unsupported file type for text extraction: {file_name}. Only PDF files are supported for local extraction.")

def extract_text_hybrid(file, use_aws: bool = True) -> str:
    """
    Hybrid text extraction: tries AWS Textract first, falls back to local extraction.
    
    Args:
        file: Uploaded file object from Streamlit
        use_aws: Whether to try AWS Textract first
        
    Returns:
        Extracted text as string
    """
    if use_aws:
        try:
            return extract_text_with_aws_textract_sync(file)
        except Exception as e:
            print(f"⚠️ AWS Textract failed for {file.name}: {e}")
            print(f"🔄 Falling back to local extraction: {file.name}")
    
    # Fallback to local extraction
    return extract_text_from_file(file)

def process_excel_file(file) -> pd.DataFrame:
    """
    Process Excel file and return structured data.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        pandas DataFrame with the Excel data
    """
    try:
        # Read Excel file
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:  # .xls
            df = pd.read_excel(file, engine='xlrd')
        
        # Basic data cleaning
        df = df.dropna(how='all')  # Remove completely empty rows
        df = df.fillna('')  # Fill NaN values with empty strings
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to process Excel file: {e}")
# app.py
import streamlit as st
import os
import pandas as pd
import io
from dotenv import load_dotenv
from utils import extract_text_hybrid, process_excel_file
from orchestrator import classify_document, process_invoice, process_bank_statement, process_general_document
import re

# --- Load Environment Variables ---
# This must be the first line of your app's code
load_dotenv() 

# --- Page Configuration ---
st.set_page_config(page_title="Document Text Extractor", page_icon="🤖", layout="wide")

# --- Configuration Check ---
# Check if essential configurations are loaded from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 Critical Error: Could not find OpenAI API Key in .env file.")
    st.stop() # Halts the app if configuration is missing

# Check AWS credentials for Textract
aws_available = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
if aws_available:
    st.success("🔒 AWS Textract available for secure text extraction")
else:
    st.warning("⚠️ AWS credentials not found - will use local text extraction only")

def validate_text_quality(text: str) -> int:
    """
    Validate the quality of extracted text for financial documents.
    Returns a quality score from 0-100.
    """
    if not text or len(text.strip()) < 10:
        return 0
    
    score = 100
    
    # Check for minimum length
    if len(text) < 50:
        score -= 30
    
    # Check for financial keywords
    financial_keywords = ['invoice', 'total', 'amount', 'vat', 'btw', 'date', 'payment', 'receipt', 'statement']
    found_keywords = sum(1 for keyword in financial_keywords if keyword.lower() in text.lower())
    if found_keywords < 3:
        score -= 20
    
    # Check for numbers (amounts, dates)
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 3:
        score -= 15
    
    # Check for currency symbols
    currency_symbols = ['€', '$', '£', 'USD', 'EUR', 'GBP']
    if not any(symbol in text for symbol in currency_symbols):
        score -= 10
    
    # Check for excessive special characters (OCR errors)
    special_chars = re.findall(r'[^\w\s€$£.,:;()\-]', text)
    if len(special_chars) > len(text) * 0.1:  # More than 10% special chars
        score -= 25
    
    # Check for repeated characters (OCR errors)
    repeated_chars = re.findall(r'(.)\1{3,}', text)  # 4+ repeated chars
    if repeated_chars:
        score -= 15
    
    return max(0, min(100, score))

# --- Main UI ---
st.title("🤖 Document Text Extractor")
st.write("Upload documents to extract text and analyze with AI")

# Initialize session state variables
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# --- File Uploader ---
st.subheader("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose files to upload", 
    type=['pdf', 'xlsx', 'xls'],
    accept_multiple_files=True,
    help="Supported formats: PDF, Excel (XLSX, XLS)"
)

# Display uploaded files info
if uploaded_files:
    st.write(f"📋 **{len(uploaded_files)} file(s) selected:**")
    for i, file in enumerate(uploaded_files):
        file_type = "📊 Excel" if file.name.endswith(('.xlsx', '.xls')) else "📄 PDF"
        st.write(f"{i+1}. {file_type} - {file.name} ({file.size:,} bytes)")

if uploaded_files:
    # --- Step 1: Extract Text from Documents ---
    if st.button("1. Extract Text from Documents"):
        with st.spinner("Extracting text from documents..."):
            try:
                extracted_texts = []
                processed_files = []
                
                for file in uploaded_files:
                    # Check if it's an Excel file
                    if file.name.endswith(('.xlsx', '.xls')):
                        st.info(f"📊 Processing Excel file: {file.name}")
                        # Process Excel file directly
                        excel_result = process_excel_file(file)
                        processed_files.append({
                            'name': file.name,
                            'type': 'excel',
                            'result': excel_result
                        })
                        st.success(f"✅ Excel file processed: {file.name}")
                    else:
                        # Extract text using hybrid approach (AWS Textract first, local fallback)
                        st.info(f"📄 Extracting text from: {file.name}")
                        extracted_text = extract_text_hybrid(file, use_aws=aws_available)
                        
                        # Validate text quality
                        quality_score = validate_text_quality(extracted_text)
                        
                        extracted_texts.append({
                            'file_name': file.name,
                            'text': extracted_text,
                            'quality_score': quality_score
                        })
                        
                        # Show quality feedback
                        if quality_score >= 80:
                            st.success(f"✅ High quality text extracted from: {file.name} (Quality: {quality_score}%)")
                        elif quality_score >= 60:
                            st.warning(f"⚠️ Medium quality text extracted from: {file.name} (Quality: {quality_score}%)")
                        else:
                            st.error(f"❌ Low quality text extracted from: {file.name} (Quality: {quality_score}%)")
                            st.info("💡 Consider using a higher quality scan or different document format")
                
                st.session_state.extracted_texts = extracted_texts
                st.session_state.processed_files = processed_files
                
                if extracted_texts:
                    st.info(f"📋 Text extracted from {len(extracted_texts)} document(s). Click below to analyze.")
                    
                    # Display extracted text
                    st.subheader("📄 Extracted Text Preview")
                    for i, result in enumerate(extracted_texts):
                        quality_score = result.get('quality_score', 0)
                        quality_color = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
                        
                        with st.expander(f"{quality_color} {result['file_name']} - Quality: {quality_score}%"):
                            st.text_area(
                                "Raw extracted text:",
                                value=result['text'],
                                height=200,
                                key=f"extracted_text_{i}",
                                disabled=True
                            )
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Character count", len(result['text']))
                            with col2:
                                st.metric("Quality score", f"{quality_score}%")
                            with col3:
                                if quality_score >= 80:
                                    st.success("High quality")
                                elif quality_score >= 60:
                                    st.warning("Medium quality")
                                else:
                                    st.error("Low quality")
                
                if processed_files:
                    st.info(f"📊 {len(processed_files)} Excel file(s) processed directly.")
                    
                    # Display Excel data
                    st.subheader("📊 Excel Data Preview")
                    for i, excel_file in enumerate(processed_files):
                        with st.expander(f"📊 {excel_file['name']} - Excel Data"):
                            st.dataframe(excel_file['result'])
                            st.caption(f"Rows: {len(excel_file['result'])}, Columns: {len(excel_file['result'].columns)}")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Step 2: Analyze and Classify Extracted Text ---
if st.session_state.extracted_texts or st.session_state.processed_files:
    st.markdown("---")
    st.subheader("2. Analyze and Classify Extracted Text")
    
    if st.button("Run AI Analysis on All Documents"):
        with st.spinner("Analyzing and classifying documents... 🧠"):
            all_results = []
            
            # Process extracted texts (PDFs, images)
            for result in st.session_state.extracted_texts:
                try:
                    file_name = result['file_name']
                    text = result['text']
                    
                    st.write(f"🔍 Analyzing: **{file_name}**")
                    
                    # Classify document
                    doc_type = classify_document(text)
                    st.info(f"📋 {file_name} classified as: **{doc_type.upper()}**")
                    
                    # Process based on type
                    if doc_type == "invoice":
                        final_output = process_invoice(text)
                    elif doc_type == "bank_statement":
                        final_output = process_bank_statement(text)
                    elif doc_type in ["tax_document", "receipt", "financial_report", "other"]:
                        final_output = process_general_document(text)
                    else:
                        st.warning(f"⚠️ {file_name}: Document type not supported")
                        continue
                    
                    all_results.append({
                        'file_name': file_name,
                        'type': doc_type,
                        'result': final_output
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing {file_name}: {e}")
            
            # Process Excel files
            for excel_file in st.session_state.processed_files:
                try:
                    file_name = excel_file['name']
                    excel_result = excel_file['result']
                    
                    st.write(f"📊 Excel file: **{file_name}**")
                    all_results.append({
                        'file_name': file_name,
                        'type': 'excel',
                        'result': excel_result
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing Excel {file_name}: {e}")
            
            # Display results
            if all_results:
                st.subheader("📋 Analysis Results")
                
                for i, result in enumerate(all_results):
                    with st.expander(f"📄 {result['file_name']} ({result['type'].upper()})"):
                        if result['type'] == 'excel':
                            st.dataframe(result['result'])
                        else:
                            st.json(result['result'].dict() if hasattr(result['result'], 'dict') else [item.dict() for item in result['result']])
                
                # Summary
                st.success(f"🎉 Successfully analyzed {len(all_results)} file(s)!")
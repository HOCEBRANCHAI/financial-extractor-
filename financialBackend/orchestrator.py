import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List

# ==============================================================================
# DOCUMENT CLASSIFIER (THE ROUTER)
# ==============================================================================
class DocumentClassifier(BaseModel):
    """Classifies a document into financial document types."""
    document_type: Literal["invoice", "bank_statement", "tax_document", "receipt", "financial_report", "other"] = Field(
        description="The type of the document. Must be one of: 'invoice', 'bank_statement', 'tax_document', 'receipt', 'financial_report', or 'other'."
    )

# ==============================================================================
# SPECIALIZED PROCESSORS (THE WORKERS)
# ==============================================================================

# --- Invoice Processor ---
class InvoiceTransaction(BaseModel):
    description: str = Field(description="Description of the line item.")
    amount_pre_vat: float = Field(description="The amount before VAT.")
    vat_percentage: int = Field(description="The VAT percentage applied.")
    vat_category: str = Field(description="The specific VAT category code.")

class InvoiceOutput(BaseModel):
    """Structured data extracted from an invoice."""
    invoice_no: str
    date: str
    invoice_to: str
    country: str
    transactions: List[InvoiceTransaction]
    total_amount: float

# --- Bank Statement Processor ---
class TransactionClassification(BaseModel):
    account_code: str = Field(description="The account code for this transaction")
    account_name: str = Field(description="The account name for this transaction")
    confidence: float = Field(description="Confidence score for the classification")

class SpecialFlags(BaseModel):
    internal_transfer: bool = Field(default=False, description="Whether this is an internal transfer")
    recurring_payment: bool = Field(default=False, description="Whether this is a recurring payment")
    tax_related: bool = Field(default=False, description="Whether this transaction is tax-related")

class BankTransaction(BaseModel):
    transaction_id: str = Field(description="Unique identifier for the transaction")
    classification: TransactionClassification = Field(description="Classification details for the transaction")
    special_flags: SpecialFlags = Field(description="Special flags for the transaction")

class BankStatementOutput(BaseModel):
    """Structured data extracted from a bank statement."""
    transactions: List[BankTransaction]

# --- General Document Processor ---
class GeneralDocumentOutput(BaseModel):
    """Structured data extracted from general financial documents."""
    document_title: str = Field(description="Title or main heading of the document")
    document_date: str = Field(description="Date mentioned in the document")
    key_amounts: List[dict] = Field(description="List of important amounts found in the document")
    key_entities: List[str] = Field(description="Important names, companies, or entities mentioned")
    summary: str = Field(description="Brief summary of the document content")

# ==============================================================================
# INITIALIZE LLM
# ==============================================================================
def get_llm():
    """Initialize and return the LLM client."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("Please set your OpenAI API key as an environment variable: set OPENAI_API_KEY=your_actual_api_key_here")
        return ChatOpenAI(model="gpt-4o", temperature=0)
    except Exception as e:
        raise Exception(f"Could not initialize OpenAI client: {e}")

# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================
def classify_document(text: str) -> str:
    """
    Analyzes raw text and returns the document type.
    """
    print(f"\n🧠 Classifying document...")
    
    llm = get_llm()
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial document classifier. Analyze the text and classify it into one of these categories:

- invoice: Bills, invoices, sales documents with line items and totals
- bank_statement: Bank account statements with transactions and balances  
- tax_document: Tax returns, VAT returns, tax forms, BTW documents
- receipt: Purchase receipts, payment confirmations
- financial_report: Financial statements, profit/loss, balance sheets
- other: Any other financial or business document

Look for keywords like: invoice, bill, statement, tax, VAT, BTW, receipt, payment, transaction, balance, etc."""),
        ("human", "Here is the document text: \n\n```{text_input}```")
    ])
    
    structured_llm_classifier = llm.with_structured_output(DocumentClassifier)
    chain = classifier_prompt | structured_llm_classifier
    
    result = chain.invoke({"text_input": text})
    print(f"✅ Document classified as: {result.document_type}")
    return result.document_type

def process_invoice(text: str) -> InvoiceOutput:
    """
    Extracts structured data from invoice text using the VAT analysis prompt.
    """
    print("⚙️ Processing as INVOICE...")
    
    llm = get_llm()
    invoice_prompt_template = """
    You are an AI assistant that extracts structured invoice data for VAT reporting from raw text.
    Extract and return only the following JSON format.

    - Extract the 'country' as the country of the client (invoice recipient).
    - For each transaction, assign a `vat_category` using ONLY these codes:
    1a → Domestic sales taxed at 21%
    1c → Sales with 0% VAT to EU countries or exports
    4b → Services purchased from EU countries

    Raw text from invoice:
    ```{text_input}```
    """
    
    prompt = ChatPromptTemplate.from_template(invoice_prompt_template)
    structured_llm_invoice = llm.with_structured_output(InvoiceOutput)
    chain = prompt | structured_llm_invoice
    
    result = chain.invoke({"text_input": text})
    print("✅ Invoice processing complete.")
    return result

def process_bank_statement(text: str) -> List[BankTransaction]:
    """
    Extracts and classifies transactions from bank statement text.
    """
    print("⚙️ Processing as BANK STATEMENT...")
    
    llm = get_llm()
    # This is a simplified version of the Agent 1 prompt for demonstration
    statement_prompt_template = """
    You are an AI accounting expert. From the raw text of a bank statement below,
    extract each transaction and classify it. The business is a software consultancy.
    - Payments to software companies are 'Kantoorkosten' (4500).
    - Payments from clients are 'Omzet Diensten' (8010).

    Return a JSON array of objects.

    Raw text from bank statement:
    ```{text_input}```
    """
    
    prompt = ChatPromptTemplate.from_template(statement_prompt_template)
    structured_llm_statement = llm.with_structured_output(BankStatementOutput)
    chain = prompt | structured_llm_statement
    
    result = chain.invoke({"text_input": text})
    print("✅ Bank statement processing complete.")
    return result.transactions

def process_general_document(text: str) -> GeneralDocumentOutput:
    """
    Extracts structured data from general financial documents.
    """
    print("⚙️ Processing as GENERAL DOCUMENT...")
    
    llm = get_llm()
    general_prompt_template = """
    You are an AI assistant that extracts structured data from financial documents.
    Analyze the document and extract key information in a structured format.

    Raw text from document:
    ```{text_input}```
    """
    
    prompt = ChatPromptTemplate.from_template(general_prompt_template)
    structured_llm_general = llm.with_structured_output(GeneralDocumentOutput)
    chain = prompt | structured_llm_general
    
    result = chain.invoke({"text_input": text})
    print("✅ General document processing complete.")
    return result

"""
Unified Newspaper Real Estate Extraction Pipeline
==================================================
This script combines DOCX extraction and structured data parsing into a single workflow.

Features:
1. Extracts raw listings from DOCX files
2. Parses and structures real estate data with regex patterns
3. Outputs both raw CSV and structured CSV files
4. Handles both sale and rental properties
5. Extracts comprehensive property details (BHK, sqft, price, location, etc.)

Usage:
    python unified_newspaper_extraction.py
"""

import csv
import pandas as pd
import re
import os
import glob
from typing import List, Tuple, Optional
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the extraction pipeline"""
    DOCX_PATH = "MTClassifiedsFeb7_13_2026-1-2.docx"
    RAW_CSV_OUTPUT = "MTClassifiedsFeb7_13_2026-1-2.csv"
    STRUCTURED_CSV_OUTPUT = "structured_real_estate_accumulated.csv"
    PROCESSED_FILES_LOG = "processed_pdfs.txt"  # Track processed PDFs
    LISTING_DELIMITER = " l "  # Delimiter between listings in DOCX
    
    # List of known localities in Chennai
    LOCALITIES = [
        "Mylapore", "Mandaveli", "Adyar", "Besant Nagar", "R.A.Puram", 
        "San Thome", "Alwarpet", "Thiruvanmiyur", "Gopalapuram", 
        "MRC Nagar", "Pallikarnai", "Velachery", "Kottivakkam", "Neelankarai"
    ]


# ============================================================================
# STEP 1: PDF & DOCX EXTRACTION
# ============================================================================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    For PDFs starting with "AT-", only extract pages containing "CLASSIFIEDS".
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Combined text from all pages (or filtered pages for AT- files)
    """
    if not PDF_AVAILABLE:
        raise ImportError("PyPDF2 is required to read PDF files. Install with: pip install PyPDF2")
    
    full_text = []
    pdf_name = os.path.basename(pdf_path)
    is_at_file = pdf_name.startswith("AT-")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    # For AT- files, only include pages with "CLASSIFIEDS" title
                    if is_at_file:
                        if "CLASSIFIEDS" in text.upper():
                            full_text.append(text.strip())
                    else:
                        full_text.append(text.strip())
    except Exception as e:
        print(f"  ⚠ Error reading {pdf_path}: {str(e)}")
        return ""
    
    return " ".join(full_text)


def find_all_pdfs(directory: str = ".") -> List[str]:
    """
    Find all PDF files in the specified directory.
    Excludes PDF files starting with "AT-" prefix.
    
    Args:
        directory: Directory to search for PDFs
        
    Returns:
        List of PDF file paths (excluding AT- files)
    """
    all_pdfs = glob.glob(os.path.join(directory, "*.pdf"))
    # Filter out AT- prefixed files
    pdf_files = [pdf for pdf in all_pdfs if not os.path.basename(pdf).startswith("AT-")]
    return sorted(pdf_files)


def extract_docx_text(docx_path: str) -> str:
    """
    Extract all text from a DOCX file.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        Combined text from all paragraphs
    """
    from docx import Document
    doc = Document(docx_path)
    full_text = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    
    return " ".join(full_text)


def split_listings(text: str, delimiter: str) -> List[str]:
    """
    Split raw text into individual listings based on delimiter.
    
    Args:
        text: Raw text from DOCX/PDF
        delimiter: String delimiter between listings
        
    Returns:
        List of individual listing strings
    """
    # Try multiple delimiter patterns common in PDFs and DOCX
    delimiters_to_try = [
        delimiter,           # Original delimiter " l "
        "\tl",              # Tab + l (common in PDFs)
        "\nl",              # Newline + l
        " \tl",             # Space + tab + l
    ]
    
    # Find which delimiter works best (gives most splits)
    best_parts = [text]
    for delim in delimiters_to_try:
        parts = text.split(delim)
        if len(parts) > len(best_parts):
            best_parts = parts
    
    listings = []
    for item in best_parts:
        cleaned = item.strip()
        if len(cleaned) > 30:  # Filter out very short fragments
            listings.append(cleaned)
    
    return listings


def get_processed_files(log_file: str) -> set:
    """
    Get set of already processed PDF files.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Set of processed filenames
    """
    if not os.path.exists(log_file):
        return set()
    
    with open(log_file, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def mark_file_as_processed(log_file: str, filename: str) -> None:
    """
    Mark a PDF file as processed.
    
    Args:
        log_file: Path to the log file
        filename: Name of the processed file
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")


def save_raw_csv(listings: List[Tuple[str, str]], output_file: str) -> None:
    """
    Save raw listings to CSV file with source filename.
    
    Args:
        listings: List of tuples (source_file, listing_text)
        output_file: Path to output CSV file
    """
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "Listing"])
        for source_file, item in listings:
            writer.writerow([source_file, item])


# ============================================================================
# STEP 2: DATA EXTRACTION WITH REGEX
# ============================================================================

def extract_phone(text: str) -> Optional[str]:
    """Extract Indian phone numbers (10 digits starting with 6-9)"""
    phones = re.findall(r"\b[6-9]\d{9}\b", text)
    return ", ".join(phones) if phones else None


def extract_bhk(text: str) -> Optional[int]:
    """Extract BHK (bedroom-hall-kitchen) count"""
    match = re.search(r"(\d+)\s*BHK", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_sqft(text: str) -> Optional[int]:
    """Extract square footage"""
    match = re.search(r"(\d{3,5})\s*sq\.?ft", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_uds(text: str) -> Optional[int]:
    """Extract UDS (Undivided Share of Land)"""
    match = re.search(r"UDS\s*(\d{2,5})", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_floor(text: str) -> Optional[str]:
    """Extract floor number"""
    match = re.search(r"(\d+)(st|nd|rd|th)\s*floor", text, re.IGNORECASE)
    return match.group(1) if match else None


def extract_facing(text: str) -> Optional[str]:
    """Extract property facing direction"""
    match = re.search(r"(east|west|north|south)\s*facing", text, re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def extract_price(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract sale price and unit (lakhs/crores).
    
    Returns:
        Tuple of (price_value, price_unit)
    """
    match = re.search(r"Rs\.?\s*([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value, unit
    return None, None


def normalize_price(value: Optional[float], unit: Optional[str]) -> Optional[int]:
    """
    Convert price to INR.
    
    Args:
        value: Numeric price value
        unit: Unit (lakhs/crores)
        
    Returns:
        Price in INR
    """
    if value is None:
        return None
    if unit and "crore" in unit:
        return int(value * 10000000)
    if unit and "lakh" in unit:
        return int(value * 100000)
    return None


def detect_rental(text: str) -> bool:
    """Detect if listing is for rent"""
    return "rent" in text.lower()


def extract_rent(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract rental amount and unit.
    
    Handles formats like: Rs. 30K, Rs 45000, Rs. 1.5L, Rs. 1.10 lakhs
    
    Returns:
        Tuple of (rent_value, rent_unit)
    """
    match = re.search(
        r"Rs\.?\s*([\d]+(?:\.\d+)?)(?:\s*(K|L|lakhs?|lakh))?",
        text,
        re.IGNORECASE
    )
    
    if not match:
        return None, None
    
    raw_value = match.group(1)
    unit = match.group(2)
    
    # Reject invalid values
    if raw_value in [".", "", None]:
        return None, None
    
    try:
        value = float(raw_value)
    except ValueError:
        return None, None
    
    return value, unit


def normalize_rent(value: Optional[float], unit: Optional[str]) -> Optional[int]:
    """
    Convert rent to INR.
    
    Args:
        value: Numeric rent value
        unit: Unit (K/L/lakhs)
        
    Returns:
        Rent in INR
    """
    if value is None:
        return None
    if unit is None:
        return int(value)
    
    unit = unit.lower()
    if unit == "k":
        return int(value * 1000)
    if unit in ["l", "lakh", "lakhs"]:
        return int(value * 100000)
    return int(value)


def extract_locality(text: str, localities: List[str]) -> Optional[str]:
    """
    Extract locality/neighborhood from text.
    
    Args:
        text: Listing text
        localities: List of known localities
        
    Returns:
        Matched locality name
    """
    for loc in localities:
        if loc.lower() in text.lower():
            return loc
    return None


def detect_property_type(text: str) -> str:
    """
    Detect property type from text.
    
    Returns:
        Property type (Land/Independent House/Apartment/Commercial/Unknown)
    """
    text_lower = text.lower()
    if "land" in text_lower or "plot" in text_lower:
        return "Land"
    if "independent" in text_lower or "house" in text_lower:
        return "Independent House"
    if "flat" in text_lower or "apartment" in text_lower:
        return "Apartment"
    if "commercial" in text_lower or "office" in text_lower:
        return "Commercial"
    return "Unknown"


# ============================================================================
# STEP 3: STRUCTURED DATA PROCESSING
# ============================================================================

def is_real_estate_listing(text: str) -> bool:
    """
    Determine if a listing is actually a real estate listing.
    
    Args:
        text: Listing text
        
    Returns:
        True if it's a real estate listing, False otherwise
    """
    text_lower = text.lower()
    
    # Keywords that indicate non-real estate content
    non_real_estate_keywords = [
        "tuition", "classes", "teacher", "education", "abacus", "handwriting",
        "phonics", "maths", "physics", "chemistry", "biology", "computer",
        "accounts", "french", "hindi", "coaching", "yoga", "dance class",
        "pest control", "termites", "bed bug", "cockroach", "sofa",
        "electronics", "inverter", "battery", "electrical services",
        "house maid", "helper", "driver", "cook", "baby sitter", "office boys",
        "patient care", "attender", "diaper changing", "manpower services",
        "food", "masala", "millet", "health mix", "adai mix", "dosai mix",
        "audio", "video", "dvd player", "blue ray"
    ]
    
    # Check if listing contains non-real estate keywords
    for keyword in non_real_estate_keywords:
        if keyword in text_lower:
            return False
    
    # Real estate indicators
    real_estate_keywords = [
        "bhk", "sq.ft", "sqft", "apartment", "flat", "house", "land", "plot",
        "rent", "sale", "buy", "property", "floor", "facing", "uds",
        "crore", "lakh", "rs.", "price", "rate", "building", "independent",
        "commercial", "residential", "ground floor", "car park", "lift"
    ]
    
    # Must have at least one real estate indicator
    has_real_estate_indicator = any(keyword in text_lower for keyword in real_estate_keywords)
    
    # Additional check: very short listings are likely not real estate
    if len(text.strip()) < 30:
        return False
    
    return has_real_estate_indicator


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and extra whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove the special character �
    text = text.replace("�", "")
    # Remove other common problematic characters
    text = text.replace("\ufffd", "")
    # Clean up extra whitespace
    text = " ".join(text.split())
    return text.strip()


def process_listings_to_structured_data(
    raw_csv_path: str, 
    output_csv_path: str,
    localities: List[str],
    append_mode: bool = False
) -> pd.DataFrame:
    """
    Process raw listings CSV into structured data.
    
    Args:
        raw_csv_path: Path to raw CSV file
        output_csv_path: Path to save structured CSV
        localities: List of known localities
        append_mode: If True, append to existing CSV instead of overwriting
        
    Returns:
        DataFrame with structured data
    """
    df = pd.read_csv(raw_csv_path)
    structured_rows = []
    skipped_count = 0
    
    for _, row in df.iterrows():
        source_file = str(row["source_file"])
        text = str(row["Listing"])
        
        # Clean the text
        text = clean_text(text)
        
        # Filter out non-real estate listings
        if not is_real_estate_listing(text):
            skipped_count += 1
            continue
        
        # Extract all fields
        bhk = extract_bhk(text)
        sqft = extract_sqft(text)
        uds = extract_uds(text)
        floor = extract_floor(text)
        facing = extract_facing(text)
        locality = extract_locality(text, localities)
        phones = extract_phone(text)
        
        # Price extraction
        price_value, price_unit = extract_price(text)
        price_in_inr = normalize_price(price_value, price_unit)
        
        # Rental extraction
        is_rental = detect_rental(text)
        rent_value, rent_unit = extract_rent(text)
        rent_in_inr = normalize_rent(rent_value, rent_unit)
        
        # Property type
        property_type = detect_property_type(text)
        
        structured_rows.append({
            "source_file": source_file,
            "listing_text": text,
            "city": "Chennai",
            "locality": locality if locality else "",
            "property_type": property_type,
            "bhk": bhk if bhk else "",
            "sqft_builtup": sqft if sqft else "",
            "sqft_uds": uds if uds else "",
            "floor": floor if floor else "",
            "facing": facing if facing else "",
            "price_value": price_value if price_value else "",
            "price_unit": price_unit if price_unit else "",
            "price_in_inr": price_in_inr if price_in_inr else "",
            "is_rental": is_rental,
            "rent_value": rent_value if rent_value else "",
            "rent_unit": rent_unit if rent_unit else "",
            "rent_in_inr": rent_in_inr if rent_in_inr else "",
            "contact_numbers": phones if phones else "",
        })
    
    new_df = pd.DataFrame(structured_rows)
    
    # Handle append mode
    if append_mode and os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        out_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    
    out_df.to_csv(output_csv_path, index=False)
    
    print(f"  → Filtered out {skipped_count} non-real estate listings")
    
    return new_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_extraction_pipeline(config: Config) -> None:
    """
    Run the complete extraction pipeline for all PDFs in the directory.
    Implements incremental processing - skips already processed files.
    
    Args:
        config: Configuration object with all settings
    """
    print("=" * 70)
    print("UNIFIED NEWSPAPER REAL ESTATE EXTRACTION PIPELINE (PDF)")
    print("=" * 70)
    
    # Step 1: Find all PDF files and check which are already processed
    print("\n[STEP 1] Finding PDF files in current directory...")
    all_pdf_files = find_all_pdfs(".")
    
    if not all_pdf_files:
        print("ERROR: No PDF files found in the current directory")
        return
    
    processed_files = get_processed_files(config.PROCESSED_FILES_LOG)
    pdf_files = [pdf for pdf in all_pdf_files if os.path.basename(pdf) not in processed_files]
    
    print(f"✓ Found {len(all_pdf_files)} PDF file(s) total")
    print(f"✓ Already processed: {len(processed_files)} file(s)")
    print(f"✓ New files to process: {len(pdf_files)} file(s)")
    
    if not pdf_files:
        print("\n✓ All PDF files have already been processed!")
        print("  Delete 'processed_pdfs.txt' to reprocess all files.")
        return
    
    print("\nNew files to process:")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")
    
    # Step 2: Extract text from new PDFs
    print("\n[STEP 2] Extracting text from new PDF files...")
    all_listings = []
    
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"  Processing: {pdf_name}")
        
        text = extract_pdf_text(pdf_path)
        if text:
            print(f"    ✓ Extracted {len(text)} characters")
            
            # Split into listings
            listings = split_listings(text, config.LISTING_DELIMITER)
            print(f"    ✓ Found {len(listings)} listings")
            
            # Add source filename to each listing
            for listing in listings:
                all_listings.append((pdf_name, listing))
        else:
            print(f"    ⚠ No text extracted from {pdf_name}")
    
    print(f"\n✓ Total listings extracted from new PDFs: {len(all_listings)}")
    
    # Step 3: Save raw CSV (temporary for this batch)
    print("\n[STEP 3] Saving raw listings to temporary CSV...")
    temp_raw_csv = "temp_raw_listings.csv"
    save_raw_csv(all_listings, temp_raw_csv)
    print(f"✓ Temporary raw CSV saved: {temp_raw_csv}")
    
    # Step 4: Process into structured data (append mode)
    print("\n[STEP 4] Processing listings into structured data...")
    append_mode = os.path.exists(config.STRUCTURED_CSV_OUTPUT)
    structured_df = process_listings_to_structured_data(
        temp_raw_csv,
        config.STRUCTURED_CSV_OUTPUT,
        config.LOCALITIES,
        append_mode=append_mode
    )
    print(f"✓ Structured CSV {'updated' if append_mode else 'created'}: {config.STRUCTURED_CSV_OUTPUT}")
    
    # Step 5: Mark files as processed
    print("\n[STEP 5] Marking files as processed...")
    for pdf_path in pdf_files:
        mark_file_as_processed(config.PROCESSED_FILES_LOG, os.path.basename(pdf_path))
    print(f"✓ Updated processing log: {config.PROCESSED_FILES_LOG}")
    
    # Clean up temporary file
    if os.path.exists(temp_raw_csv):
        os.remove(temp_raw_csv)
    
    # Step 6: Summary statistics
    print("\n[STEP 6] Summary Statistics:")
    print("-" * 70)
    print(f"New PDFs processed: {len(pdf_files)}")
    print(f"New listings added: {len(structured_df)}")
    print(f"Rental properties: {structured_df['is_rental'].sum()}")
    print(f"Sale properties: {(~structured_df['is_rental']).sum()}")
    print(f"Properties with BHK info: {(structured_df['bhk'] != '').sum()}")
    print(f"Properties with sqft info: {(structured_df['sqft_builtup'] != '').sum()}")
    print(f"Properties with locality info: {(structured_df['locality'] != '').sum()}")
    print(f"Properties with contact info: {(structured_df['contact_numbers'] != '').sum()}")
    
    # Show total accumulated statistics
    if append_mode:
        print("\nTotal Accumulated Statistics:")
        print("-" * 70)
        total_df = pd.read_csv(config.STRUCTURED_CSV_OUTPUT)
        print(f"Total listings in database: {len(total_df)}")
        print(f"Total rental properties: {total_df['is_rental'].sum()}")
        print(f"Total sale properties: {(~total_df['is_rental']).sum()}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = Config()
    run_extraction_pipeline(config)

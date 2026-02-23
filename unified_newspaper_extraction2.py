"""
Unified Newspaper Real Estate Extraction Pipeline (PDF + OCR)
=============================================================

- Handles text-based and scanned PDFs (Mylapore Times style)
- Extracts classifieds, splits into listings, filters real estate
- Parses BHK, sqft, UDS, facing, floor, price, rent, locality, phones
- Appends to a single structured CSV with incremental processing
"""

import os
import re
import csv
import glob
from typing import List, Tuple, Optional

import pandas as pd

import pdfplumber
import pytesseract
from PIL import Image


# ======================================================================
# CONFIG
# ======================================================================

class Config:
    PDF_GLOB = "MTClassifieds*.pdf"          # pattern for your newspaper PDFs
    RAW_CSV_OUTPUT = "raw_listings_temp.csv"
    STRUCTURED_CSV_OUTPUT = "structured_real_estate_accumulated.csv"
    PROCESSED_FILES_LOG = "processed_pdfs.txt"

    LOCALITIES = [
        "Mylapore", "Mandaveli", "Adyar", "Besant Nagar", "R.A.Puram",
        "San Thome", "Alwarpet", "Thiruvanmiyur", "Gopalapuram",
        "MRC Nagar", "Pallikarnai", "Velachery", "Kottivakkam", "Neelankarai", "Abhiramapuram", "CIT Colony"
    ]


# ======================================================================
# PDF DISCOVERY & LOGGING
# ======================================================================

def find_all_pdfs(pattern: str) -> List[str]:
    pdfs = glob.glob(pattern)
    return sorted(pdfs)


def get_processed_files(log_file: str) -> set:
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_file_as_processed(log_file: str, filename: str) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


# ======================================================================
# PDF TEXT EXTRACTION (TEXT + OCR)
# ======================================================================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from each page.
    - If text layer exists → use it
    - If not → OCR the page image
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            # If page has almost no text, fall back to OCR
            if len(page_text) < 30:
                img = page.to_image(resolution=300).original
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                ocr_text = pytesseract.image_to_string(img)
                page_text = ocr_text.strip()

            if page_text:
                all_text.append(page_text)

    return "\n".join(all_text)


# ======================================================================
# LISTING SPLITTING
# ======================================================================

def split_listings(raw_text: str) -> List[str]:
    """
    Split classifieds text into individual listings.
    Uses bullet characters and line heuristics.
    """
    # Normalize bullets
    text = raw_text.replace("•", "·").replace("●", "·").replace("▪", "·").replace("‣", "·")

    # First, split by newlines
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    listings = []
    current = []

    def flush_current():
        if current:
            joined = " ".join(current).strip()
            if len(joined) > 30:
                listings.append(joined)
            current.clear()

    for line in lines:
        # Start of a new listing if:
        # - line starts with bullet or dot
        # - or looks like a typical classified start (ALL CAPS locality + comma)
        if re.match(r"^[·\-\*\.]\s+", line) or re.match(r"^[A-Z][A-Z\s\.\-]+,", line):
            flush_current()
            # remove leading bullet/dot
            line = re.sub(r"^[·\-\*\.]\s*", "", line)
            current.append(line)
        else:
            current.append(line)

    flush_current()
    return listings


def save_raw_csv(listings: List[Tuple[str, str]], output_file: str) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "Listing"])
        for src, text in listings:
            writer.writerow([src, text])


# ======================================================================
# FIELD EXTRACTION HELPERS
# ======================================================================

def clean_text(text: str) -> str:
    text = text.replace("�", "").replace("\ufffd", "")
    return " ".join(text.split()).strip()


def extract_phone(text: str) -> Optional[str]:
    phones = re.findall(r"\b[6-9]\d{9}\b", text)
    return ", ".join(sorted(set(phones))) if phones else None


def extract_bhk(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*BHK", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_sqft(text: str) -> Optional[int]:
    m = re.search(r"(\d{3,5})\s*sq\.? ?ft", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_uds(text: str) -> Optional[int]:
    m = re.search(r"UDS\s*(\d{2,5})", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def extract_floor(text: str) -> Optional[str]:
    m = re.search(r"(\d+)(st|nd|rd|th)\s*floor", text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_facing(text: str) -> Optional[str]:
    m = re.search(r"(east|west|north|south)\s*facing", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def extract_price(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Sale price: look for 'Rate' or 'Price' or 'Rs.' with lakhs/crores.
    """
    # Prefer patterns with 'Rate' or 'Price'
    m = re.search(r"(Rate|Price)\s*([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Rs\.?\s*([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        return None, None
    return float(m.group(2 if m.lastindex >= 2 else 1)), m.group(m.lastindex)


def normalize_price(value: Optional[float], unit: Optional[str]) -> Optional[int]:
    if value is None or not unit:
        return None
    unit = unit.lower()
    if "crore" in unit:
        return int(value * 10_000_000)
    if "lakh" in unit:
        return int(value * 100_000)
    return None


def detect_rental(text: str) -> bool:
    t = text.lower()
    return "rent" in t or "rental" in t or "lease" in t


def extract_rent(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Rent: Rs. 30K, Rs 45000, Rs. 1.5L, Rs. 1.10 lakhs, etc.
    """
    # Focus on segments near 'rent'
    snippet = text
    m_rent_word = re.search(r"rent[^.,;]*", text, re.IGNORECASE)
    if m_rent_word:
        snippet = m_rent_word.group(0)

    m = re.search(r"Rs\.?\s*([\d]+(?:\.\d+)?)(?:\s*(K|L|lakhs?|lakh))?", snippet, re.IGNORECASE)
    if not m:
        return None, None

    raw = m.group(1)
    unit = m.group(2)
    try:
        val = float(raw)
    except ValueError:
        return None, None
    return val, unit


def normalize_rent(value: Optional[float], unit: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if not unit:
        return int(value)
    unit = unit.lower()
    if unit == "k":
        return int(value * 1000)
    if unit in ("l", "lakh", "lakhs"):
        return int(value * 100_000)
    return int(value)


def extract_locality(text: str, localities: List[str]) -> Optional[str]:
    t = text.lower()
    for loc in localities:
        if loc.lower() in t:
            return loc
    return None


def detect_property_type(text: str) -> str:
    t = text.lower()
    if "plot" in t or "land" in t:
        return "Land"
    if "independent" in t or "house" in t or "bungalow" in t:
        return "Independent House"
    if "flat" in t or "apartment" in t:
        return "Apartment"
    if "commercial" in t or "office" in t or "shop" in t:
        return "Commercial"
    return "Unknown"


def is_real_estate_listing(text: str) -> bool:
    t = text.lower()

    non_real_estate = [
        "tuition", "classes", "teacher", "education", "coaching",
        "pest control", "sofa", "manpower", "house maid", "cook",
        "baby sitter", "driver", "nurse", "caretaker", "beauty parlour",
        "hotel business", "music classes", "dance classes", "matrimonial",
        "alliance", "change of name", "name as per", "clinic", "ayurveda",
    ]
    if any(k in t for k in non_real_estate):
        return False

    real_estate = [
        "bhk", "sq.ft", "sqft", "apartment", "flat", "house", "land", "plot",
        "rent", "lease", "sale", "rate", "price", "uds", "car park", "lift",
        "ground floor", "independent", "bungalow", "gated community"
    ]
    if not any(k in t for k in real_estate):
        return False

    return len(text.strip()) > 30


# ======================================================================
# STRUCTURED PROCESSING
# ======================================================================

def process_listings_to_structured(
    raw_csv_path: str,
    output_csv_path: str,
    localities: List[str],
    append_mode: bool = False
) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        src = str(row["source_file"])
        text = clean_text(str(row["Listing"]))

        if not is_real_estate_listing(text):
            skipped += 1
            continue

        bhk = extract_bhk(text)
        sqft = extract_sqft(text)
        uds = extract_uds(text)
        floor = extract_floor(text)
        facing = extract_facing(text)
        locality = extract_locality(text, localities)
        phones = extract_phone(text)

        is_rent = detect_rental(text)
        price_val, price_unit = (None, None)
        price_in_inr = None
        rent_val, rent_unit, rent_in_inr = (None, None, None)

        if is_rent:
            rent_val, rent_unit = extract_rent(text)
            rent_in_inr = normalize_rent(rent_val, rent_unit)
        else:
            price_val, price_unit = extract_price(text)
            price_in_inr = normalize_price(price_val, price_unit)

        prop_type = detect_property_type(text)

        rows.append({
            "source_file": src,
            "listing_text": text,
            "city": "Chennai",
            "locality": locality or "",
            "property_type": prop_type,
            "bhk": bhk or "",
            "sqft_builtup": sqft or "",
            "sqft_uds": uds or "",
            "floor": floor or "",
            "facing": facing or "",
            "price_value": price_val or "",
            "price_unit": price_unit or "",
            "price_in_inr": price_in_inr or "",
            "is_rental": is_rent,
            "rent_value": rent_val or "",
            "rent_unit": rent_unit or "",
            "rent_in_inr": rent_in_inr or "",
            "contact_numbers": phones or "",
        })

    new_df = pd.DataFrame(rows)

    if append_mode and os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        out_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(output_csv_path, index=False)

    print(f"  → Skipped {skipped} non-real-estate listings")
    return new_df


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_pipeline(config: Config) -> None:
    print("=" * 70)
    print("UNIFIED REAL ESTATE EXTRACTION (PDF + OCR)")
    print("=" * 70)

    all_pdfs = find_all_pdfs(config.PDF_GLOB)
    if not all_pdfs:
        print("No PDFs found matching pattern:", config.PDF_GLOB)
        return

    processed = get_processed_files(config.PROCESSED_FILES_LOG)
    new_pdfs = [p for p in all_pdfs if os.path.basename(p) not in processed]

    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Already processed: {len(processed)}")
    print(f"New to process: {len(new_pdfs)}")

    if not new_pdfs:
        print("Nothing new to process.")
        return

    all_listings = []

    for pdf_path in new_pdfs:
        name = os.path.basename(pdf_path)
        print(f"\n[PDF] {name}")
        text = extract_pdf_text(pdf_path)
        print(f"  Extracted {len(text)} characters")

        listings = split_listings(text)
        print(f"  Found {len(listings)} candidate listings")

        for lst in listings:
            all_listings.append((name, lst))

    print(f"\nTotal listings from new PDFs: {len(all_listings)}")

    save_raw_csv(all_listings, config.RAW_CSV_OUTPUT)
    print(f"Raw listings saved to: {config.RAW_CSV_OUTPUT}")

    append_mode = os.path.exists(config.STRUCTURED_CSV_OUTPUT)
    structured_df = process_listings_to_structured(
        config.RAW_CSV_OUTPUT,
        config.STRUCTURED_CSV_OUTPUT,
        config.LOCALITIES,
        append_mode=append_mode,
    )

    print(f"\nStructured CSV {'updated' if append_mode else 'created'}: {config.STRUCTURED_CSV_OUTPUT}")
    print(f"New structured rows: {len(structured_df)}")

    for pdf_path in new_pdfs:
        mark_file_as_processed(config.PROCESSED_FILES_LOG, os.path.basename(pdf_path))
    print(f"Processing log updated: {config.PROCESSED_FILES_LOG}")

    if os.path.exists(config.RAW_CSV_OUTPUT):
        os.remove(config.RAW_CSV_OUTPUT)

    print("\nSummary:")
    print("-" * 70)
    print(f"New PDFs processed: {len(new_pdfs)}")
    print(f"New listings stored: {len(structured_df)}")
    print(f"Rental: {structured_df['is_rental'].sum()}")
    print(f"Sale: {(~structured_df['is_rental']).sum()}")
    print(f"With BHK: {(structured_df['bhk'] != '').sum()}")
    print(f"With sqft: {(structured_df['sqft_builtup'] != '').sum()}")
    print(f"With locality: {(structured_df['locality'] != '').sum()}")
    print(f"With contact: {(structured_df['contact_numbers'] != '').sum()}")
    print("=" * 70)


if __name__ == "__main__":
    cfg = Config()
    run_pipeline(cfg)

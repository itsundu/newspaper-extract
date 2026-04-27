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
    PDF_GLOB = r".\mylaporetimes\MTClassifieds*.pdf"
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

def _words_to_column_text(words: list, page_width: float) -> str:
    """Reconstruct reading-order text from word objects, respecting multi-column layout."""
    if not words:
        return ""

    # Estimate line height from word bounding boxes
    heights = [w['bottom'] - w['top'] for w in words if w['bottom'] > w['top']]
    line_h = max(6.0, (sum(heights) / len(heights)) * 0.65) if heights else 9.0

    # Build x-position histogram to locate column separators
    n_bins = 80
    bin_size = page_width / n_bins
    hist = [0] * n_bins
    for w in words:
        b = min(int(w['x0'] / bin_size), n_bins - 1)
        hist[b] += 1

    # A real column gap must be at least 2 consecutive empty bins (~1.5% page width)
    # to avoid false splits on word spacing within a column.
    col_starts = [0.0]
    i = 0
    while i < n_bins:
        if hist[i] == 0:
            j = i
            while j < n_bins and hist[j] == 0:
                j += 1
            if (j - i) >= 2 and j < n_bins:
                col_starts.append(j * bin_size)
            i = j
        else:
            i += 1
    col_starts.append(page_width)

    def get_col(x):
        for ci in range(len(col_starts) - 1):
            if col_starts[ci] <= x < col_starts[ci + 1]:
                return ci
        return len(col_starts) - 2

    # Bucket each word by (column_index, row_index)
    buckets: dict = {}
    for w in words:
        key = (get_col(w['x0']), int(w['top'] / line_h))
        buckets.setdefault(key, []).append(w)

    # Emit lines: column by column, top-to-bottom within each column
    lines = []
    for key in sorted(buckets):
        row_words = sorted(buckets[key], key=lambda w: w['x0'])
        lines.append(' '.join(w['text'] for w in row_words))

    return '\n'.join(lines)


def _looks_garbled(text: str) -> bool:
    """Return True if text shows signs of multi-column mixing."""
    words = [w for w in text.split() if w.isalpha()]
    if not words:
        return False
    # Many single-char words → character-level interleaving
    single = sum(1 for w in words if len(w) == 1)
    if (single / len(words)) > 0.20:
        return True
    # Multiple bullets on same line → row-across-columns mixing (layout=True artifact)
    lines = [l for l in text.split('\n') if l.strip()]
    multi_bullet_lines = sum(
        1 for l in lines
        if l.count('•') + l.count('·') + l.count('●') > 1
    )
    return multi_bullet_lines > 2


def _ocr_page(page) -> str:
    """Render page to image and run Tesseract OCR. Returns empty string if unavailable."""
    try:
        img = page.to_image(resolution=300).original
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from each page with multi-column awareness.
    Strategy (in order):
      1. pdfplumber layout mode (pdfminer LAParams) — handles columns natively
      2. Word-position column reconstruction — fallback when layout mode unavailable
      3. OCR — last resort for scanned/image pages (skipped if Tesseract absent)
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = ""

            # --- Strategy 1: layout-aware extraction (pdfplumber >= 0.9) ---
            try:
                page_text = page.extract_text(layout=True) or ""
                page_text = page_text.strip()
            except TypeError:
                # Older pdfplumber doesn't support layout=True
                page_text = ""

            # --- Strategy 2: word-position column reconstruction ---
            if len(page_text) < 30 or _looks_garbled(page_text):
                words = page.extract_words(x_tolerance=7, y_tolerance=3)
                page_text = _words_to_column_text(words, page.width) if words else ""

            # --- Strategy 3: OCR (only if text layer is truly absent) ---
            if len(page_text.strip()) < 30:
                page_text = _ocr_page(page)

            if page_text:
                all_text.append(page_text)

    full_text = "\n".join(all_text)
    full_text = re.sub(
        r'YOU CAN READ THE CLASSIFIEDS ONLINE[^\n]*',
        '',
        full_text,
        flags=re.IGNORECASE
    )
    return full_text


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

    # When layout=True outputs multiple columns on the same line, bullets end up
    # mid-line.  Force each bullet onto its own line so the line-based logic below
    # correctly treats it as a new listing start.
    text = re.sub(r'([^\n])(·)', r'\1\n\2', text)

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
    # Require at least one leading digit so a bare '.' never matches
    m = re.search(r"(Rate|Price)\s*(\d[\d\.]*)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Rs\.?\s*(\d[\d\.]*)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d[\d\.]*)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not m:
        return None, None
    # Pattern 1 has 3 groups (keyword, number, unit); patterns 2 & 3 have 2 (number, unit)
    num_group = 2 if m.lastindex == 3 else 1
    try:
        return float(m.group(num_group)), m.group(m.lastindex)
    except ValueError:
        return None, None


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
    return bool(re.search(r'\bRent\b', text, re.IGNORECASE) or re.search(r'\blease\b', text, re.IGNORECASE))


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


def extract_price_simple(text: str) -> Optional[int]:
    """Fallback: extract price from NNNK (e.g. 90K) or Rs. NNNNN format."""
    # Check K-suffix first so "Rs. 90K" returns 90000 not 90
    m = re.search(r'\b(\d[\d\.]*)\s*K\b', text)
    if m:
        try:
            return int(float(m.group(1)) * 1000)
        except ValueError:
            pass
    m = re.search(r'Rs\.?\s*(\d[\d,]*(?:\.\d+)?)\b', text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(',', '')
        try:
            return int(float(val_str))
        except ValueError:
            pass
    return None


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

    # These phrases indicate the listing is NOT about property — only block when
    # they appear without any real-estate signal words in the same text.
    non_real_estate = [
        "pest control", "manpower", "house maid", "baby sitter",
        "beauty parlour", "matrimonial", "alliance",
        "change of name", "name as per", "ayurveda",
        "tuition centre", "coaching centre", "dance academy", "music academy",
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
            price_in_inr = rent_in_inr
            rent_val = rent_in_inr
        else:
            price_val, price_unit = extract_price(text)
            price_in_inr = normalize_price(price_val, price_unit)

        if price_in_inr is None:
            price_in_inr = extract_price_simple(text)

        # For rentals, keep rent_in_inr in sync with the final price_in_inr value
        if is_rent and price_in_inr is not None:
            rent_in_inr = price_in_inr

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

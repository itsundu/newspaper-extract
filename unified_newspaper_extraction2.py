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
import requests

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

    # Supabase sync: reads SUPABASE_URL / SUPABASE_SERVICE_KEY from the
    # environment (GitHub Actions secrets in CI, a local .env otherwise).
    # The push is skipped with a warning if either is unset, so the script
    # still runs standalone (CSV-only) without Supabase configured.
    SUPABASE_TABLE = "active_listings"


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

def _detect_column_bounds(words: list, page_width: float, page_height: float) -> List[float]:
    """
    Detect stable column gutters using per-band coverage voting.

    A single page-wide coverage histogram is fooled by a handful of full-width
    lines (mastheads, rule lines, section headers): those lines fill in the
    gutter bins for the whole page and erase an otherwise-consistent column
    boundary. Instead, coverage is computed per horizontal band and a bin only
    counts as a gutter if it is empty in most of the bands that actually have
    content -- so a couple of outlier full-width lines can't destroy a real
    column boundary.
    """
    if not words:
        return [0.0, page_width]

    n_bins = 200
    bin_size = page_width / n_bins
    band_h = 50.0
    n_bands = max(1, int(page_height / band_h) + 1)

    empty_votes = [0] * n_bins
    active_bands = 0

    for bi in range(n_bands):
        y0, y1 = bi * band_h, (bi + 1) * band_h
        band_words = [w for w in words if y0 <= w['top'] < y1]
        if not band_words:
            continue
        active_bands += 1
        covered = [False] * n_bins
        for w in band_words:
            b0 = max(0, min(n_bins - 1, int(w['x0'] / bin_size)))
            b1 = max(0, min(n_bins - 1, int(w['x1'] / bin_size)))
            for b in range(b0, b1 + 1):
                covered[b] = True
        for b in range(n_bins):
            if not covered[b]:
                empty_votes[b] += 1

    if active_bands == 0:
        return [0.0, page_width]

    threshold = active_bands * 0.6
    is_gutter_bin = [empty_votes[b] >= threshold for b in range(n_bins)]

    min_gutter_bins = max(1, int(n_bins * 0.01))
    col_starts = [0.0]
    i = 0
    while i < n_bins:
        if is_gutter_bin[i]:
            j = i
            while j < n_bins and is_gutter_bin[j]:
                j += 1
            if (j - i) >= min_gutter_bins and 0 < j < n_bins:
                col_starts.append(((i + j) / 2) * bin_size)
            i = j
        else:
            i += 1
    col_starts.append(page_width)
    return col_starts


def _join_line_words(row_words: list) -> str:
    """
    Join words sorted left-to-right into a line of text.

    Some PDFs (this newspaper's export included) break a single word like
    "UDS" into separate one-character word objects at kerning-pair
    boundaries, with ~0pt gaps between them, while real word-to-word gaps
    measure >=0.8pt even at small font sizes. Unconditionally joining every
    word with a literal space turns "UDS 680" into "U D S 6 8 0"; instead,
    only insert a space where there's an actual visual gap.
    """
    parts = []
    prev_x1 = None
    for w in row_words:
        if prev_x1 is not None and (w['x0'] - prev_x1) > 0.6:
            parts.append(' ')
        parts.append(w['text'])
        prev_x1 = w['x1']
    return ''.join(parts)


def _words_to_column_text(words: list, page_width: float, page_height: float) -> str:
    """Reconstruct reading-order text from word objects, respecting multi-column layout."""
    if not words:
        return ""

    col_starts = _detect_column_bounds(words, page_width, page_height)

    def get_col(x):
        for ci in range(len(col_starts) - 1):
            if col_starts[ci] <= x < col_starts[ci + 1]:
                return ci
        return len(col_starts) - 2

    columns: dict = {}
    for w in words:
        columns.setdefault(get_col(w['x0']), []).append(w)

    # Within each column, cluster words into lines by comparing each word's
    # top to a running reference (not a fixed-width bucket grid) -- a fixed
    # grid can split one physical line across two buckets when normal glyph
    # jitter straddles a bucket boundary, then re-merge fragments of
    # adjacent lines, producing character-interleaved output.
    lines = []
    for col_idx in sorted(columns):
        col_words = sorted(columns[col_idx], key=lambda w: (w['top'], w['x0']))
        current_line: list = []
        current_top = None
        for w in col_words:
            tol = max(2.0, (w['bottom'] - w['top']) * 0.5)
            if current_top is not None and (w['top'] - current_top) > tol:
                current_line.sort(key=lambda ww: ww['x0'])
                lines.append(_join_line_words(current_line))
                current_line = []
                current_top = None
            current_line.append(w)
            current_top = w['top'] if current_top is None else min(current_top, w['top'])
        if current_line:
            current_line.sort(key=lambda ww: ww['x0'])
            lines.append(_join_line_words(current_line))

    return '\n'.join(lines)


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
      1. Word-position column reconstruction — band-voted gutter detection,
         robust to the irregular multi-column classifieds layout (see
         _detect_column_bounds). This is the primary strategy: pdfplumber's
         layout=True mode zips adjacent columns' text together character-by-
         character on pages with tightly packed multi-column text.
      2. pdfplumber layout mode (pdfminer LAParams) — fallback when word
         extraction is unavailable or empty.
      3. OCR — last resort for scanned/image pages (skipped if Tesseract absent)
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = ""

            # --- Strategy 1: word-position column reconstruction ---
            words = page.extract_words(x_tolerance=7, y_tolerance=3)
            if words:
                page_text = _words_to_column_text(words, page.width, page.height).strip()

            # --- Strategy 2: layout-aware extraction (pdfplumber >= 0.9) ---
            if len(page_text) < 30:
                try:
                    page_text = (page.extract_text(layout=True) or "").strip()
                except TypeError:
                    # Older pdfplumber doesn't support layout=True
                    page_text = page_text

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

    # Some issues' font encoding renders the bullet glyph as a lone lowercase
    # "l" (not OCR noise -- the surrounding text extracts cleanly). Treat a
    # line-leading "l " the same as the other bullet characters; no real
    # classified line starts with the single-letter word "l".
    bullet_start = re.compile(r"^([·\-\*\.]|l)\s+")

    for line in lines:
        # Start of a new listing if:
        # - line starts with bullet or dot
        # - or looks like a typical classified start (ALL CAPS locality + comma)
        if bullet_start.match(line) or re.match(r"^[A-Z][A-Z\s\.\-]+,", line):
            flush_current()
            # remove leading bullet/dot
            line = bullet_start.sub("", line)
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
# MULTI-PROPERTY LISTING SPLITTING
# ======================================================================

def looks_like_multi_property(text: str) -> bool:
    """True if a listing mentions sq.ft or UDS more than once -- a sign that
    more than one property is bundled into the same bullet/contact number."""
    sqft_count = len(re.findall(r"sq\.?\s*ft", text, re.IGNORECASE))
    uds_count = len(re.findall(r"\bUDS\b", text, re.IGNORECASE))
    return sqft_count >= 2 or uds_count >= 2


def split_multi_property_listing(text: str, localities: List[str]) -> List[str]:
    """
    Some bulleted classifieds advertise more than one property (often in
    different localities) under a single shared phone number, e.g.:
      "MANDAVELI, ... Rate 2.25 crores, Besant Nagar, ... Rate 3.25 crores. Ph: ..."
    Split such listings into one segment per locality mention, so each
    property gets its own row, with the shared phone number re-attached to
    every segment that doesn't already carry it.
    """
    if not looks_like_multi_property(text):
        return [text]

    loc_pattern = re.compile(
        r"\b(" + "|".join(re.escape(l) for l in localities) + r")\b",
        re.IGNORECASE,
    )
    matches = list(loc_pattern.finditer(text))
    if len(matches) < 2:
        return [text]

    # Collapse locality matches that are really the same mention repeated
    # close together (e.g. "Mandaveli, Mandaveli Extension").
    starts = [matches[0].start()]
    for m in matches[1:]:
        if m.start() - starts[-1] > 15:
            starts.append(m.start())
    if len(starts) < 2:
        return [text]

    segments = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[s:e].strip(" ,.")
        if seg:
            segments.append(seg)
    if len(segments) < 2:
        return [text]

    # Preserve any bullet-intro text before the first locality mention.
    if starts[0] > 0:
        prefix = text[: starts[0]].strip(" ,.")
        if prefix:
            segments[0] = f"{prefix} {segments[0]}"

    phones = extract_phone(text)
    if phones:
        for i, seg in enumerate(segments):
            if not extract_phone(seg):
                segments[i] = f"{seg.rstrip('.')} Ph: {phones}"

    return segments


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

    split_count = 0

    for _, row in df.iterrows():
        src = str(row["source_file"])
        text = clean_text(str(row["Listing"]))

        if not is_real_estate_listing(text):
            skipped += 1
            continue

        parts = split_multi_property_listing(text, localities)
        if len(parts) > 1:
            split_count += 1

        for part in parts:
            bhk = extract_bhk(part)
            sqft = extract_sqft(part)
            uds = extract_uds(part)
            floor = extract_floor(part)
            facing = extract_facing(part)
            locality = extract_locality(part, localities)
            phones = extract_phone(part)

            is_rent = detect_rental(part)
            price_val, price_unit = (None, None)
            price_in_inr = None
            rent_val, rent_unit, rent_in_inr = (None, None, None)

            if is_rent:
                rent_val, rent_unit = extract_rent(part)
                rent_in_inr = normalize_rent(rent_val, rent_unit)
                price_in_inr = rent_in_inr
            else:
                price_val, price_unit = extract_price(part)
                price_in_inr = normalize_price(price_val, price_unit)

            if price_in_inr is None:
                price_in_inr = extract_price_simple(part)

            # For rentals, keep rent_in_inr in sync with the final price_in_inr value
            if is_rent and price_in_inr is not None:
                rent_in_inr = price_in_inr

            prop_type = detect_property_type(part)

            rows.append({
                "source_file": src,
                "listing_text": part,
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
    print(f"  → Split {split_count} multi-property listings into separate rows")
    return new_df


# ======================================================================
# SUPABASE SYNC
# ======================================================================

def _to_supabase_row(row: dict) -> dict:
    """Convert one structured-CSV row into a JSON-safe payload matching the
    active_listings schema (every column is `text` except is_rental)."""
    out = {}
    for key, value in row.items():
        if key == "is_rental":
            out[key] = bool(value)
        elif value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
            out[key] = None
        else:
            out[key] = str(value)
    return out


def push_new_rows_to_supabase(df: pd.DataFrame, table: str) -> None:
    """
    Insert newly extracted rows into the Supabase active_listings table.

    Reads SUPABASE_URL / SUPABASE_SERVICE_KEY from the environment (set as
    GitHub Actions secrets in CI). The service role key is required because
    active_listings has no RLS policy allowing anon inserts; it must never be
    committed to the repo. If either variable is unset, the push is skipped
    with a warning so the script still works standalone.
    """
    if df.empty:
        print("  → No new rows to push to Supabase")
        return

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("  → SUPABASE_URL / SUPABASE_SERVICE_KEY not set; skipping Supabase push")
        return

    endpoint = url.rstrip("/") + f"/rest/v1/{table}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    payload_rows = [_to_supabase_row(r) for r in df.to_dict(orient="records")]

    batch_size = 500
    for i in range(0, len(payload_rows), batch_size):
        batch = payload_rows[i:i + batch_size]
        resp = requests.post(endpoint, headers=headers, json=batch, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Supabase insert failed ({resp.status_code}): {resp.text[:500]}"
            )

    print(f"  → Pushed {len(payload_rows)} new rows to Supabase ({table})")


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

    # Push before marking PDFs as processed: if this raises, the log is left
    # untouched so the same PDFs are retried (and re-pushed) on the next run
    # instead of silently losing their rows from Supabase.
    push_new_rows_to_supabase(structured_df, config.SUPABASE_TABLE)

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

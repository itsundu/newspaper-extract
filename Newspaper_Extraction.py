import csv
from docx import Document
import os

# ---------------------------------------
# CONFIG
# ---------------------------------------
DOCX_PATH = "MTClassifiedsFeb7_13_2026-1-2.docx"
OUTPUT_CSV = "MTClassifiedsFeb7_13_2026-1-2.csv"
DELIMITER = " l "   # delimiter between listings

# ---------------------------------------
# READ DOCX TEXT
# ---------------------------------------
def extract_docx_text(docx_path):
    doc = Document(docx_path)
    full_text = []

    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    return " ".join(full_text)

# ---------------------------------------
# SPLIT LISTINGS
# ---------------------------------------
def split_listings(text, delimiter):
    parts = text.split(delimiter)
    listings = []

    for item in parts:
        cleaned = item.strip()
        if len(cleaned) > 3:
            listings.append(cleaned)

    return listings

# ---------------------------------------
# SAVE TO CSV
# ---------------------------------------
def save_to_csv(listings, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Listing"])
        for item in listings:
            writer.writerow([item])

# ---------------------------------------
# MAIN
# ---------------------------------------
if __name__ == "__main__":
    print("Reading DOCX file...")
    text = extract_docx_text(DOCX_PATH)

    print("Splitting listings...")
    listings = split_listings(text, DELIMITER)

    print(f"Found {len(listings)} listings. Saving to CSV...")
    save_to_csv(listings, OUTPUT_CSV)

    print("Done! CSV created:", OUTPUT_CSV)

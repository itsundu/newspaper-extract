import pandas as pd
import re

INPUT_CSV = "MTClassifiedsFeb7_13_2026-1-2.csv"
OUTPUT_CSV = "structured_real_estate_accumulated.csv"

# ---------------------------------------------------------
# REGEX HELPERS
# ---------------------------------------------------------

def extract_phone(text):
    phones = re.findall(r"\b[6-9]\d{9}\b", text)
    return ", ".join(phones) if phones else None

def extract_bhk(text):
    match = re.search(r"(\d+)\s*BHK", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_sqft(text):
    match = re.search(r"(\d{3,5})\s*sq\.?ft", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_uds(text):
    match = re.search(r"UDS\s*(\d{2,5})", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def extract_floor(text):
    match = re.search(r"(\d+)(st|nd|rd|th)\s*floor", text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_facing(text):
    match = re.search(r"(east|west|north|south)\s*facing", text, re.IGNORECASE)
    return match.group(1).capitalize() if match else None

def extract_price(text):
    match = re.search(r"Rs\.?\s*([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"([\d\.]+)\s*(lakhs?|crores?)", text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value, unit
    return None, None

def normalize_price(value, unit):
    if value is None:
        return None
    if "crore" in unit:
        return int(value * 10000000)
    if "lakh" in unit:
        return int(value * 100000)
    return None

def detect_rental(text):
    return "rent" in text.lower() or "rs." in text.lower() and "rent" in text.lower()

def extract_rent(text):
    # Match patterns like:
    # Rs. 30K, Rs 45000, Rs. 1.5L, Rs. 1.10 lakhs, Rs 1 lakh
    match = re.search(
        r"Rs\.?\s*([\d]+(?:\.\d+)?)(?:\s*(K|L|lakhs?|lakh))?",
        text,
        re.IGNORECASE
    )

    if not match:
        return None, None

    raw_value = match.group(1)
    unit = match.group(2)

    # Reject invalid values like "."
    if raw_value in [".", "", None]:
        return None, None

    try:
        value = float(raw_value)
    except ValueError:
        return None, None

    return value, unit


def normalize_rent(value, unit):
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

def extract_locality(text):
    localities = [
        "Mylapore","Mandaveli","Adyar","Besant Nagar","R.A.Puram","San Thome",
        "Alwarpet","Thiruvanmiyur","Gopalapuram","MRC Nagar","Pallikarnai",
        "Velachery","Kottivakkam","Neelankarai"
    ]
    for loc in localities:
        if loc.lower() in text.lower():
            return loc
    return None

def detect_property_type(text):
    if "land" in text.lower() or "plot" in text.lower():
        return "Land"
    if "independent" in text.lower() or "house" in text.lower():
        return "Independent House"
    if "flat" in text.lower() or "apartment" in text.lower():
        return "Apartment"
    if "commercial" in text.lower() or "office" in text.lower():
        return "Commercial"
    return "Unknown"

# ---------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

structured_rows = []

for _, row in df.iterrows():
    text = str(row["Listing"])

    bhk = extract_bhk(text)
    sqft = extract_sqft(text)
    uds = extract_uds(text)
    floor = extract_floor(text)
    facing = extract_facing(text)
    locality = extract_locality(text)
    phones = extract_phone(text)

    price_value, price_unit = extract_price(text)
    price_in_inr = normalize_price(price_value, price_unit)

    is_rental = detect_rental(text)
    rent_value, rent_unit = extract_rent(text)
    rent_in_inr = normalize_rent(rent_value, rent_unit)

    property_type = detect_property_type(text)

    structured_rows.append({
        "listing_text": text,
        "city": "Chennai",
        "locality": locality,
        "property_type": property_type,
        "bhk": bhk,
        "sqft_builtup": sqft,
        "sqft_uds": uds,
        "floor": floor,
        "facing": facing,
        "price_value": price_value,
        "price_unit": price_unit,
        "price_in_inr": price_in_inr,
        "is_rental": is_rental,
        "rent_value": rent_value,
        "rent_unit": rent_unit,
        "rent_in_inr": rent_in_inr,
        "contact_numbers": phones,
    })

out_df = pd.DataFrame(structured_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print("Structured dataset saved to:", OUTPUT_CSV)

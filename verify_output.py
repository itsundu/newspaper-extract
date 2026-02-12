import pandas as pd

df = pd.read_csv('structured_real_estate_accumulated.csv')

print(f'Total rows: {len(df)}')
print(f'\nSource files:')
print(df['source_file'].value_counts())
print(f'\n\nSample listings:')
print(df[['source_file', 'locality', 'property_type', 'bhk', 'sqft_builtup', 'price_in_inr', 'is_rental']].head(10).to_string())
print(f'\n\nListing text samples (first 3):')
for i in range(min(3, len(df))):
    print(f"\n--- Listing {i+1} ---")
    print(f"Source: {df.iloc[i]['source_file']}")
    print(f"Text: {df.iloc[i]['listing_text'][:200]}...")

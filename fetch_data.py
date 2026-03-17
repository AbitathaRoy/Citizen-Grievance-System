# fetch_data.py

import pandas as pd
import os

API_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

def fetch_stratified_sample():
    print("Initiating 12-month stratified data pull to eliminate temporal bias...")
    all_data = []
    
    # Loop through all 12 months of 2025
    for month in range(1, 13):
        # Format dates for SoQL (e.g., '2025-01-01T00:00:00.000')
        start_date = f"2025-{str(month).zfill(2)}-01T00:00:00.000"
        end_date = f"2025-{str(month).zfill(2)}-28T23:59:59.000" # Using 28th to avoid leap year/month-end math
        
        print(f"Pulling 1,000 records for Month {month}...")
        
        query_params = {
            "$limit": 1000,
            "$select": "created_date, agency, complaint_type, descriptor, resolution_description",
            "$where": f"descriptor IS NOT NULL AND created_date >= '{start_date}' AND created_date <= '{end_date}'"
        }
        
        url = f"{API_ENDPOINT}?$limit={query_params['$limit']}&$select={query_params['$select']}&$where={query_params['$where']}"
        url = url.replace(" ", "%20")
        
        try:
            df_month = pd.read_csv(url)
            all_data.append(df_month)
        except Exception as e:
            print(f"Failed to fetch data for month {month}: {e}")
            
    # Combine all months
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle the dataset randomly using a fixed random state for reproducibility
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    file_path = "data/nyc_311_shuffled_sample.csv"
    shuffled_df.to_csv(file_path, index=False)
    
    print(f"\nSuccess. Saved {len(shuffled_df)} randomized rows to {file_path}.")
    return shuffled_df

if __name__ == "__main__":
    df = fetch_stratified_sample()
    print("\nRandomized Data Snapshot:")
    print(df[['agency', 'complaint_type', 'descriptor']].head(10))
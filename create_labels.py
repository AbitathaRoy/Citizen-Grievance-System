# create_labels.py

import os
import time
import json
import pandas as pd
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant" 
# MODEL = "llama-3.3-70b-versatile"
# MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL = "qwen/qwen3-32b"
# MODEL = "openai/gpt-oss-120b"
# MODEL "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are an expert civic triage AI. Analyze the civic complaint data provided. 
Calculate an urgency score based on immediate risk to public safety, property damage, or severe disruption.
Output ONLY a valid JSON object with exactly two keys:
1. "sentiment_label": Choose one from ["Neutral", "Frustrated", "Angry", "Critical Panic"]
2. "urgency_score": An integer from 1 to 10 (1 = trivial, 10 = immediate emergency)
"""

def generate_labels_with_cache(input_csv, output_csv, target_sample_size=100):
    print(f"Loading raw data from {input_csv}...")
    df_raw = pd.read_csv(input_csv)
    
    processed_ids = set()

    # 1. Check for existing cache
    if os.path.exists(output_csv):
        df_cache = pd.read_csv(output_csv)
        if 'original_index' in df_cache.columns:
            processed_ids = set(df_cache['original_index'].astype(str).tolist())
            print(f"Cache found! {len(processed_ids)} rows already processed.")
    else:
        # Create the output file with headers if it doesn't exist
        print("No cache found. Initializing new output file...")
        headers = ['original_index'] + list(df_raw.columns) + ['sentiment_label', 'urgency_score']
        pd.DataFrame(columns=headers).to_csv(output_csv, index=False)

    if len(processed_ids) >= target_sample_size:
        print(f"Target of {target_sample_size} rows already met. Nothing to do.")
        return

    print(f"Resuming execution to reach {target_sample_size} rows...")

    # 2. Slice the raw data up to our target size
    df_target = df_raw.head(target_sample_size)

    # 3. Process missing rows
    for index, row in df_target.iterrows():
        unique_id = str(row['created_date'])
        if unique_id in processed_ids:
            continue # Skip cached rows

        complaint_text = f"Complaint Type: {row['complaint_type']}. Descriptor: {row['descriptor']}. Resolution: {row.get('resolution_description', 'None')}"

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": complaint_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1 
            )

            result = json.loads(response.choices[0].message.content)
            sentiment = result.get("sentiment_label", "Neutral")
            urgency = result.get("urgency_score", 1)

            print(f"Index {index} | {row['complaint_type']} -> Score: {urgency} ({sentiment})")

            # 4. Append directly to disk immediately (Bulletproof caching)
            row_dict = row.to_dict()
            row_dict['original_index'] = unique_id
            row_dict['sentiment_label'] = sentiment
            row_dict['urgency_score'] = urgency

            pd.DataFrame([row_dict]).to_csv(output_csv, mode='a', header=False, index=False)

            time.sleep(1.5) # Throttle

        except Exception as e:
            print(f"API Error at index {index}: {e}")
            print("Pausing for 10 seconds before retrying...")
            time.sleep(10)

    print(f"\nExecution finished. Data safely stored in {output_csv}")

if __name__ == "__main__":
    INPUT_FILE = "data/nyc_311_shuffled_sample.csv"
    OUTPUT_FILE = "data/nyc_311_labelled_training_data.csv"
    
    # Run with a small sample first. 
    # If it crashes at row 25, running it again will instantly skip to row 26.
    generate_labels_with_cache(INPUT_FILE, OUTPUT_FILE, target_sample_size=12000)
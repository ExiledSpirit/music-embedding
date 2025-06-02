import os
import argparse
import pandas as pd
import requests
import numpy as np
import soundfile as sf
from io import BytesIO
import openl3

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="Download and embed Spotify preview tracks.")
parser.add_argument("--limit", type=int, default=10, help="Number of rows to process (default: 10)")
args = parser.parse_args()

# --- Setup ---
output_folder = "C:\\Users\\Exile\\songs"
os.makedirs(output_folder, exist_ok=True)

csv_path = "./dataset.csv"
df = pd.read_csv(csv_path)

# --- Select N rows ---
df_subset = df.head(args.limit).copy()
embeddings = []

# --- Process ---
for idx, row in df_subset.iterrows():
    url = row.get("spotify_track_preview_url")

    if pd.isna(url):
        print(f"[{idx}] No URL found, skipping.")
        embeddings.append(None)
        continue

    try:
        print(f"[{idx}] Downloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Read audio
        audio_data, sample_rate = sf.read(BytesIO(response.content), dtype='float32')
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Save local copy (optional)
        wav_filename = os.path.join(output_folder, f"track_{idx}.wav")
        sf.write(wav_filename, audio_data, sample_rate)

        # Compute embedding
        emb, _ = openl3.get_audio_embedding(audio_data, sample_rate, input_repr="mel256", content_type="music", embedding_size=512)
        avg_emb = np.mean(emb, axis=0)
        embeddings.append(",".join(map(str, avg_emb)))

        print(f"[{idx}] Embedded successfully.")

    except Exception as e:
        print(f"[{idx}] Failed: {e}")
        embeddings.append(None)

# --- Save Output ---
df_subset["openl3_embedding"] = embeddings
df_subset.to_csv("dataset_with_embeddings.csv", index=False)
print(f"Saved {len(df_subset)} rows to dataset_with_embeddings.csv")

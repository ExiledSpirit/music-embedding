import os
import pandas as pd
import requests
import numpy as np
import soundfile as sf
from io import BytesIO

output_folder = "C:\\Users\\Aluno\\songs"
os.makedirs(output_folder, exist_ok=True)

csv_path = "./dataset.csv"  # Replace with your actual CSV file path
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():

    if (idx < 362): continue
    url = row.get("spotify_track_preview_url")

    if pd.isna(url):
        print(f"[{idx}] No URL found, skipping.")
        continue

    try:
        print(f"[{idx}] Downloading from {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Load audio data into a NumPy array
        audio_data, sample_rate = sf.read(BytesIO(response.content), dtype='float32')

        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Set output filename
        wav_filename = os.path.join(output_folder, f"track_{idx}.wav")

        # Save as WAV file
        sf.write(wav_filename, audio_data, sample_rate)
        print(f"[{idx}] Saved mono WAV to {wav_filename}")

    except Exception as e:
        print(f"[{idx}] Failed to process URL: {e}")

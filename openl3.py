import os
import pandas as pd
import requests
from pydub import AudioSegment
from io import BytesIO

output_folder = "downloads"
os.makedirs(output_folder, exist_ok=True)

csv_path = "your_file.csv" # Replace with your actual CSV file path
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    url = row.get("spotify_track_preview_url")

    if pd.isna(url):
        print(f"[{idx}] No URL found, skipping.")
        continue

try:
    print(f"[{idx}] Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    # Load MP3 data into AudioSegment
    mp3_audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")

    # Set output filename
    wav_filename = os.path.join(output_folder, f"track_{idx}.wav")

    # Export as WAV
    mp3_audio.export(wav_filename, format="wav")
    print(f"[{idx}] Saved to {wav_filename}")

except Exception as e:
    print(f"[{idx}] Failed to process URL: {e}")
import os
import subprocess
import pandas as pd
from pathlib import Path
from accurate_audio_features import extract_advanced_features  # IMPORTAÇÃO AQUI

def download_song_youtube(song_name, download_dir="downloads"):
    os.makedirs(download_dir, exist_ok=True)
    safe_name = "".join(c for c in song_name if c.isalnum() or c in " _-").rstrip()
    output_path = os.path.join(download_dir, f"{safe_name}.mp3")

    command = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--quiet",
        "--no-warnings",
        "--ffmpeg-location", "ffmpeg",
        "--output", output_path,
        f"ytsearch1:{song_name}"
    ]

    subprocess.run(command)
    return output_path if os.path.exists(output_path) else None

def process_csv_songs(csv_path, output_csv="results_from_csv.csv", limit=20):
    df = pd.read_csv(csv_path)
    df = df[df['SongID'].notna()].head(limit)

    results = []
    for _, row in df.iterrows():
        song_name = row['SongID']
        file_path = download_song_youtube(song_name)

        expected_key = row.get('key')
        expected_mode = row.get('mode')
        expected_loudness = row.get('loudness')
        expected_tempo = row.get('tempo')
        expected_time_signature = row.get('time_signature')

        if file_path:
            try:
                pred_key, pred_mode, pred_loudness, pred_tempo, pred_time_signature = extract_advanced_features(file_path)
                results.append({
                    "song": song_name,
                    "expected_key": expected_key,
                    "expected_mode": expected_mode,
                    "expected_loudness": expected_loudness,
                    "expected_tempo": expected_tempo,
                    "expected_time_signature": expected_time_signature,
                    "pred_key": pred_key,
                    "pred_mode": pred_mode,
                    "pred_loudness": pred_loudness,
                    "pred_tempo": pred_tempo,
                    "pred_time_signature": pred_time_signature
                })
            except Exception as e:
                print(f"⚠️ Erro ao processar {song_name}: {e}")
        else:
            print(f"❌ Não foi possível baixar: {song_name}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Resultados exportados para: {output_csv}")

# === Execução
if __name__ == "__main__":
    process_csv_songs("../dataset_with_embeddings.csv")

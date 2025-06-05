import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
import openl3
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# --- Logging ---
def log(msg, prefix="ℹ️"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix} {msg}")

# --- CLI Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=10, help="Number of songs to process")
args = parser.parse_args()

# --- Paths ---
csv_path = "./dataset.csv"
output_csv = "dataset_with_embeddings.csv"
songs_folder = "songs"
os.makedirs(songs_folder, exist_ok=True)

# --- Load Dataset ---
df = pd.read_csv(csv_path)
df_subset = df.head(args.limit).copy()

# --- Resume Support ---
if os.path.exists(output_csv):
    log("Resuming from existing CSV...", prefix="🔁")
    prev_df = pd.read_csv(output_csv)
    if "openl3_embedding" in prev_df.columns:
        df_subset["openl3_embedding"] = prev_df["openl3_embedding"]
    else:
        df_subset["openl3_embedding"] = [None] * len(df_subset)
else:
    df_subset["openl3_embedding"] = [None] * len(df_subset)

# --- GPU Info ---
gpus = tf.config.list_physical_devices("GPU")
log(f"TensorFlow version: {tf.__version__}")
log(f"GPU detected: {bool(gpus)}", prefix="✅" if gpus else "❌")

# --- Load OpenL3 model once ---
log("Loading OpenL3 model...", prefix="🎧")
model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="music", embedding_size=512
)

# --- Processing function ---
def process_file(idx):
    try:
        t0 = time.time()
        wav_path = os.path.join(songs_folder, f"track_{idx}.wav")

        if not os.path.exists(wav_path):
            return idx, None, f"WAV not found: {wav_path}", None

        audio_data, sr = sf.read(wav_path, dtype='float32')
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        t1 = time.time()

        emb, _ = openl3.get_audio_embedding(audio_data, sr, model=model)
        avg_emb = np.mean(emb, axis=0)
        t2 = time.time()

        return idx, ",".join(map(str, avg_emb)), None, (t1 - t0, t2 - t1)

    except Exception as e:
        return idx, None, str(e), None

# --- Execution ---
log("Starting embedding from existing WAV files...", prefix="🚀")
save_interval = 10
processed_count = 0
to_process = []

# Build task list
for idx, row in df_subset.iterrows():
    if pd.notna(row.get("openl3_embedding")) and str(row["openl3_embedding"]).strip() != "":
        continue
    to_process.append(idx)

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_file, idx): idx for idx in to_process}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
        idx, emb_str, error, times = future.result()

        if error:
            log(f"[{idx}] ❌ Failed: {error}")
        else:
            dl_time, embed_time = times
            df_subset.at[idx, "openl3_embedding"] = emb_str
            log(f"[{idx}] ✅ Embedded. ⏳ Load: {dl_time:.2f}s | Embed: {embed_time:.2f}s", prefix="📥")

        processed_count += 1
        if processed_count % save_interval == 0:
            log(f"Saving checkpoint at {processed_count} songs...", prefix="💾")
            df_subset.to_csv(output_csv, index=False)

# --- Final Save ---
log("Saving final CSV...", prefix="💾")
df_subset.to_csv(output_csv, index=False)
log("✅ All done. Saved to dataset_with_embeddings.csv")

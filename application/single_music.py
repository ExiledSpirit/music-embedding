import os
import numpy as np
import soundfile as sf
import openl3
import tensorflow as tf
from recommend import recommend_song

tf.config.threading.set_intra_op_parallelism_threads(1)


import gc

model = openl3.models.load_audio_embedding_model(
    input_repr="mel256", content_type="music", embedding_size=512
)


def generate_embedding(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio_data, sr = sf.read(input_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    embedding, _ = openl3.get_audio_embedding(audio_data, sr, model=model)
    embedding_mean = np.mean(embedding, axis=0)

    np.save(output_path, embedding_mean)

    del audio_data, embedding
    gc.collect()
    tf.keras.backend.clear_session()

    return output_path

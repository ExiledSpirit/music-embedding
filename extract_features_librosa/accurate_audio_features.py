import numpy as np
import pyloudnorm as pyln
import librosa
from essentia.standard import MonoLoader, KeyExtractor
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

def extract_audio_features_accurate(file_path):
    # === LOAD AUDIO
    y, sr = librosa.load(file_path, sr=44100)
    audio = y.astype(np.float32)

    # === LOUDNESS
    meter = pyln.Meter(sr)  # BS.1770
    loudness = meter.integrated_loudness(audio)

    # === TEMPO via madmom
    proc = RNNBeatProcessor()(file_path)
    tempo = DBNBeatTrackingProcessor(fps=100)(proc)
    bpm_estimate = 60.0 / np.median(np.diff(tempo)) if len(tempo) > 1 else 0

    # === KEY + MODE via Essentia
    key_extractor = KeyExtractor()
    key_str, scale, strength = key_extractor(audio)
    key_note = key_str.split(':')[0]
    mode = 1 if scale == "major" else 0

    # === TIME SIGNATURE (heuristic estimation)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo_librosa, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    ts_estimate = 4
    if len(beats) >= 4:
        intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
        avg_interval = np.mean(intervals)
        if avg_interval < 0.3:
            ts_estimate = 6
        elif avg_interval < 0.6:
            ts_estimate = 4
        else:
            ts_estimate = 3

    return key_note, mode, loudness, bpm_estimate, ts_estimate

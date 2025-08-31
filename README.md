# Music Feature Analysis Pipeline

This project provides a complete pipeline for **music feature extraction**, **embedding**, and **prediction** of Spotify-like features using both traditional signal processing and deep learning techniques.

---

## Project Structure

### [`/embedding`](./embedding)
- **Purpose:** Extracts **deep audio embeddings** from songs using [OpenL3](https://github.com/marl/openl3).
- **Tech:** Self-supervised neural network trained on audio-video co-occurrence.
- **Usage:** Converts raw audio (WAV/MP3) into 512-dimensional embedding vectors representing high-level musical semantics (genre, mood, instrumentation, etc.).
- **Output:** CSV or `.npy` files containing OpenL3 embeddings for each song.

---

### [`/extract_features_librosa`](./extract_features_librosa)
- **Purpose:** Extracts **standard musical features** using signal processing techniques (no ML).
- **Features Extracted:**
  - Key, Mode
  - Loudness (via RMS & dB)
  - Tempo (BPM)
  - Time Signature (estimated)
- **Libs used:**
  - `librosa` for basic audio analysis
  - `essentia` for pitch/key estimation
  - `madmom` for tempo and beat tracking (multi-level beat detection supported)

---

### [`/prediction_models`](./prediction_models)
- **Purpose:** Contains the machine learning models used to predict Spotify features from OpenL3 embeddings.
- **Files:**
  - `danceability_model.py`: Trains regressors for a single target (e.g., danceability).
  - `multioutput.py`: Trains multi-output models for predicting multiple audio features at once.
- **Models Used:** Random Forest, XGBoost, MultiOutputRegressor, GridSearchCV, etc.

---

### [`/model_evaluation`](./model_evaluation)
- **Purpose:** Evaluates trained models using metrics such as:
  - `R²` (coefficient of determination)
  - `MSE` (mean squared error)
- **Supports:** Automated feature selection, multiple target evaluation, and test/train split metrics visualization.

---

## Technologies & Libraries

- [OpenL3](https://github.com/marl/openl3) – Deep audio embedding
- [XGBoost](https://xgboost.readthedocs.io) – Gradient boosting for regression
- [Scikit-learn](https://scikit-learn.org) – Model training and evaluation
- [Librosa](https://librosa.org) – Traditional audio signal processing
- [Essentia](https://essentia.upf.edu) – Music analysis and feature extraction
- [Madmom](https://github.com/CPJKU/madmom) – Beat tracking and rhythm analysis

---

## Goals

- Predict Spotify audio features from raw audio using embeddings and/or direct analysis.
- Compare performance of signal-based and embedding-based methods.
- Generate accurate and explainable results for:
  - Danceability
  - Energy
  - Loudness
  - Tempo
  - Key & Mode
  - Valence and more

---

## Example Pipeline

1. Download audio files (`.mp3`, `.wav`)
2. Extract OpenL3 embeddings → `/embedding`
3. Extract traditional features (tempo, key, etc.) → `/extract_features_librosa`
4. Train ML models → `/prediction_models`
5. Evaluate results → `/model_evaluation`

---

## Requirements

Make sure to install dependencies for:
- `librosa`, `openl3`, `essentia`, `madmom`, `scikit-learn`, `xgboost`
- FFmpeg and Python 3.9+ (for compatibility with madmom/essentia)

See the `env.yml` file for Conda-based setup instructions.

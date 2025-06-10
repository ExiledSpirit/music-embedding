import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = joblib.load('trained_music_model.joblib')
X = np.load('embeddings.npy')
df = pd.read_csv('metadata_with_embeddings.csv')

def recommend_song(input_embedding, n=5, min_danceability=0.6):
    features = model.predict(input_embedding.reshape(1, -1))[0]    
    mask = (df['danceability'] >= min_danceability) if min_danceability else None    
    sim_scores = cosine_similarity(input_embedding.reshape(1, -1), X).flatten()
    
    if mask is not None:
        sim_scores[~mask] = -1 
    
    top_indices = np.argsort(sim_scores)[-n-1:-1][::-1]
    return df.iloc[top_indices][["Song", "Performer", "spotify_track_preview_url"]]



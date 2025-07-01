import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import ast

# === Configuração ===
MODEL_PATH = "../trained_music_model.joblib"   # Altere para o nome correto do seu arquivo
CSV_PATH = "../dataset_with_embeddings.csv"         # CSV com as mesmas features
FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence"
]

# === Carregamento dos dados e embeddings ===
df = pd.read_csv(CSV_PATH)
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

X = np.stack(df['embedding'].values)
y = df[FEATURES].values

# === Separar conjunto de teste (mesmo seed) ===
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Carregar modelo salvo ===
model = joblib.load(MODEL_PATH)

# === Predição ===
y_pred = model.predict(X_test)

# === Plot: Real vs Previsto para cada feature ===
num_features = len(FEATURES)
cols = 4
rows = int(np.ceil(num_features / cols))
plt.figure(figsize=(cols * 4, rows * 4))

for i, feature in enumerate(FEATURES):
    plt.subplot(rows, cols, i + 1)
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.4, s=10)
    plt.plot([y_test[:, i].min(), y_test[:, i].max()],
             [y_test[:, i].min(), y_test[:, i].max()],
             'r--', label='Ideal')
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    plt.title(f"{feature}\nR² = {r2:.2f}")
    plt.xlabel("Real")
    plt.ylabel("Previsto")
    plt.tight_layout()

plt.suptitle("Avaliação do Modelo: Real vs Previsto por Feature", fontsize=16, y=1.02)
plt.show()

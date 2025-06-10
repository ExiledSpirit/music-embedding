import pandas as pd
import numpy as np
import ast
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Load CSV
# =========================
csv_path = "../dataset_with_embeddings.csv"

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Failed to read CSV: {e}")
    exit(1)

if 'openl3_embedding' not in df.columns:
    print("'openl3_embedding' column not found.")
    exit(1)

# =========================
# Preprocessamento dos Embeddings
# =========================
print("Removendo linhas sem embeddings")
df = df[df['openl3_embedding'].notna()].copy()

def parse_embedding(embedding_str):
    try:
        return np.array(ast.literal_eval(embedding_str))
    except Exception:
        return None

df['embedding'] = df['openl3_embedding'].apply(parse_embedding)
df = df[df['embedding'].notna()].copy()

# =========================
# Input & Output Features
# =========================
X = np.stack(df['embedding'].values)
print(f"Embedding shape: {X.shape}")

target_cols = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

missing_targets = [col for col in target_cols if col not in df.columns]
if missing_targets:
    print(f"Missing target columns: {missing_targets}")
    exit(1)

y = df[target_cols].values
print(f"Target shape: {y.shape}")

# =========================
# Split dados
# =========================
print("Splitting datasets test/treino...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Treina o modelo
# =========================
print("treinando o modelo")
start = time.time()

model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
)
model.fit(X_train, y_train)

duration = time.time() - start
print(f"Training complete in {duration:.2f} seconds")

# =========================
# testes
# =========================
print("avaliando o modelo")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

results = pd.DataFrame({
    'Feature': target_cols,
    'R2 Score': r2,
    'MSE': mse
})
print("Performance:\n", results)

results.to_csv("model_performance.csv", index=False)

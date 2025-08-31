import pandas as pd
import numpy as np
import ast
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === CONFIG ===
MODEL_PATH = "../trained_music_model.joblib"   # Altere para o nome correto do seu arquivo
CSV_PATH = "../dataset_with_embeddings.csv"         # CSV com as mesmas features
FEATURES = [
    "danceability", "energy", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence"
]

# === Load data and embeddings ===
df = pd.read_csv(CSV_PATH)
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
X = np.stack(df['embedding'].values)

# Ensure features exist
missing = [col for col in FEATURES if col not in df.columns]
if missing:
    raise ValueError(f"Missing target columns in CSV: {missing}")
y = df[FEATURES].values

# === Split (same seed used during training) ===
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Load model ===
model = joblib.load(MODEL_PATH)

# === Predict and evaluate ===
y_pred = model.predict(X_test)

# === Calculate MSE and R² for each feature ===
results = []
for i, feature in enumerate(FEATURES):
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    results.append({"Feature": feature, "MSE": mse, "R2 Score": r2})

results_df = pd.DataFrame(results).sort_values(by="MSE", ascending=False)
print("y_test valence (amostra):", y_test[:, FEATURES.index("valence")][:10])
print("y_pred valence (amostra):", y_pred[:, FEATURES.index("valence")][:10])

print(results_df)

# === Plot MSE bar chart ===
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="MSE", y="Feature", palette="Reds_r")
plt.title("Erro Médio Quadrático (MSE) por Feature")
plt.xlabel("MSE (quanto menor, melhor)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
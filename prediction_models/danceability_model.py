import pandas as pd
import numpy as np
import ast
import itertools
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# === CONFIG ===
csv_path = "../dataset_with_embeddings.csv"
output_txt = "feature_combinations_results.txt"
target_features = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]
extra_candidates = ["tempo", "key", "mode"]

# === Load Data
df = pd.read_csv(csv_path)
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
embedding_array = np.stack(df['embedding'].values)

# === Prepare combinations
feature_combinations = []
for i in range(0, 4):  # 0 to 3 features
    for combo in itertools.combinations(extra_candidates, i):
        feature_combinations.append(list(combo))

# === Run tests
with open(output_txt, "w") as f:
    for target in target_features:
        f.write(f"\n==== TARGET: {target} ====\n")
        y = df[target].values

        for combo in feature_combinations:
            # Build feature matrix
            if combo:
                extra = df[combo].values
                X = np.hstack([embedding_array, extra])
            else:
                X = embedding_array

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model
            model = XGBRegressor(
                tree_method="hist",
                device="cuda",
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            f.write(f"Features: {combo or 'None'}\n")
            f.write(f"  R²: {r2:.4f}\n")
            f.write(f"  MSE: {mse:.6f}\n\n")

print(f"Arquivo gerado: {output_txt}")

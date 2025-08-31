import pandas as pd
import numpy as np
import ast
import itertools
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# === CONFIG ===
csv_path = "../dataset_with_embeddings.csv"
output_txt = "feature_combinations_grid_results.txt"

# Removidos 'tempo' e 'loudness' dos targets
target_features = [
    "danceability", "energy",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence"
]

# Adicionados 'loudness' e 'time_signature' como features auxiliares
extra_candidates = ["tempo", "key", "mode", "loudness", "time_signature"]

param_grid = {
    "n_estimators": [100],
    "max_depth": [5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

# === Load Data
df = pd.read_csv(csv_path)
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
embedding_array = np.stack(df['embedding'].values)

# === Prepare combinations of 0 a 3 features auxiliares
feature_combinations = []
for i in range(0, min(4, len(extra_candidates) + 1)):  # safe bound
    feature_combinations.extend(itertools.combinations(extra_candidates, i))

# === Run tests
with open(output_txt, "w") as f:
    for target in target_features:
        f.write(f"\n==== TARGET: {target} ====\n")
        y = df[target].values

        for combo in feature_combinations:
            combo = list(combo)
            try:
                # Cria X com ou sem features auxiliares
                if combo:
                    extra = df[combo].values
                    X = np.hstack([embedding_array, extra])
                else:
                    X = embedding_array

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                base_model = XGBRegressor(
                    tree_method="hist",
                    device="cuda",
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1
                )

                grid = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=3,
                    scoring='r2',
                    verbose=0,
                    n_jobs=1
                )

                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                f.write(f"Features: {combo or 'None'}\n")
                f.write(f"  Best Params: {grid.best_params_}\n")
                f.write(f"  R²: {r2:.4f}\n")
                f.write(f"  MSE: {mse:.6f}\n\n")
            except Exception as e:
                f.write(f"Features: {combo or 'None'}\n")
                f.write(f"  ERRO: {e}\n\n")

print(f"[✓] Resultados com GridSearch salvos em: {output_txt}")

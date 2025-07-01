import pandas as pd
import numpy as np
import ast
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# === Load data
df = pd.read_csv("../dataset_with_embeddings.csv")
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
X = np.stack(df['embedding'].values)
y = df['danceability'].values

# === Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Reduce for tuning
X_sample = X_train[:2000]
y_sample = y_train[:2000]

# === Param grid (smaller)
param_dist = {
    "n_estimators": [100],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# === Model (GPU optimized)
model = XGBRegressor(
    tree_method="hist",  # for GPU with XGBoost >=2.0
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# === Randomized Search
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=6,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# === Fit on subset
search.fit(X_sample, y_sample)

# === Final model on full data with best params
final_model = XGBRegressor(
    **search.best_params_,
    tree_method="hist",
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

# === Evaluate
y_pred = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Melhor combinação encontrada:", search.best_params_)
print("R²:", round(r2, 4))
print("MSE:", round(mse, 6))

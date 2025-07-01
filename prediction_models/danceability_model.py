import pandas as pd
import numpy as np
import ast
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# === Load and preprocess
df = pd.read_csv("../dataset_with_embeddings.csv")
df = df[df['openl3_embedding'].notna()].copy()

# Parse embedding
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

# Combine embedding with tempo, key, mode
extra_features = df[['tempo', 'key', 'mode']].values
X = np.hstack([np.stack(df['embedding'].values), extra_features])

# Target
y = df['danceability'].values

# === Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model config
model = XGBRegressor(
    tree_method="hist",
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# Reduced grid to avoid long execution time
param_grid = {
    "n_estimators": [100],
    "max_depth": [5],
    "learning_rate": [0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

# === Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# === Train
grid.fit(X_train, y_train)

# === Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Melhor combinação:", grid.best_params_)
print("R²:", round(r2, 4))
print("MSE:", round(mse, 6))

import pandas as pd
import numpy as np
import ast
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# === Load Data
df = pd.read_csv("../dataset_with_embeddings.csv")
df = df[df['openl3_embedding'].notna()].copy()
df['embedding'] = df['openl3_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
X = np.stack(df['embedding'].values)

# === Output variables: danceability, tempo, key, mode
target_columns = ['danceability', 'tempo', 'key', 'mode']
y = df[target_columns].values

# === Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Param Grid (moderadamente complexo, mas leve)
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [5, 6],
    "learning_rate": [0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [1, 3],
    "gamma": [0, 0.1],
}

# === XGBoost Regressor com GPU
model = XGBRegressor(
    tree_method="hist",   # XGBoost 2.0+ compatível com GPU via device='cuda'
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)

# === Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=1
)

# === Train
grid.fit(X_train, y_train)

# === Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mse = mean_squared_error(y_test, y_pred)

print("Melhor combinação:", grid.best_params_)
print("R²:", round(r2, 4))
print("MSE:", round(mse, 6))

# === Opcional: Early stopping com o melhor modelo
# (útil para avaliação mais estável, pode ajustar ainda mais)
final_model = XGBRegressor(
    **grid.best_params_,
    tree_method="hist",
    device="cuda",
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

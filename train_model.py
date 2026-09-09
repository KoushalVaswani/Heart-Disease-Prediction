import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("heart(1).csv")

# Same cleaning as the notebook: 0 is a placeholder for "missing" in these two columns
for col in ["Cholesterol", "RestingBP"]:
    mean_val = df.loc[df[col] != 0, col].mean()
    df[col] = df[col].replace(0, mean_val).round(2)

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numerical_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# small grid search instead of guessing k, since we're rebuilding anyway
param_grid = {
    "model__n_neighbors": [5, 7, 9, 11, 13, 15],
    "model__weights": ["uniform", "distance"],
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
y_pred = best_pipeline.predict(X_test)

print("Best params:", grid.best_params_)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

joblib.dump(best_pipeline, "heart_pipeline.pkl")
print("\nSaved heart_pipeline.pkl (preprocessing + model bundled together)")
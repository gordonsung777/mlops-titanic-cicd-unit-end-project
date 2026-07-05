import json
from pathlib import Path 
import joblib 
import mlflow 
import mlflow.sklearn 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import (
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 

DATA_PATH = Path("data/raw/titanic.csv")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1 
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["HasCabin"] = df["Cabin"].notna().astype(int)
df["Title"] = df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
common_titles = ["Mr", "Mrs", "Miss", "Master"]
df["Title"] = df["Title"].where(df["Title"].isin(common_titles), "Rare")
fare_cap = df["Fare"].quantile(0.99)
df["Fare"] = df["Fare"].clip(upper=fare_cap)

target = "Survived"

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
]

categorical_features = [
    "Sex",
    "Embarked",
    "Title",
]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42, 
    stratify=y
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6, 
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)



#mlflow.set_tracking_uri("file:mruns")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-readmission-risk")
with mlflow.start_run(run_name="random-forest-model"):
    mlflow.log_param("algorithm", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.20)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:,1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision":precision_score(y_test, y_pred),
        "recall":recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probability),
    }
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=2)
    
    #mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["numpy.dtype"],
    )
    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(METRICS_PATH))

print("\nTraining complete.")
print("Saved model to:", MODEL_PATH)
print("Saved metrics to:", METRICS_PATH)
print(metrics)
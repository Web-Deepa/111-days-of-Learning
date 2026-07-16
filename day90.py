# day 90: MLflow 
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 1. Load and split data
data = load_breast_cancer()
x, y = data.data, data.target
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

# 2. Set experiment name local database setup
mlflow.set_experiment("breast_cancer_classification")

# 3. Setup configurations 
configs = [
    {"model": RandomForestClassifier(n_estimators=50,  random_state=42), "name": "RF_50"},
    {"model": RandomForestClassifier(n_estimators=100, random_state=42), "name": "RF_100"},
    {"model": GradientBoostingClassifier(n_estimators=100, random_state=42), "name": "GBM"},
    {"model": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), "name": "LR"},
]

best_acc = 0
best_run = None

for config in configs:
    # Start the experiment run context
    with mlflow.start_run(run_name=config['name']) as run:
        model = config['model']
        model.fit(x_tr, y_tr)
        
        y_pred = model.predict(x_te)
        y_prob = model.predict_proba(x_te)[:, 1]
        
        # Safely extract internal estimator info across Pipelines and Base Estimators
        base_model = model.steps[-1][1] if hasattr(model, 'steps') else model
        mlflow.log_param("model_type", type(base_model).__name__)
        mlflow.log_param("n_estimators", getattr(base_model, 'n_estimators', 'N/A'))
    
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc",  auc)

        # Log model artifact safely
        mlflow.sklearn.log_model(model, "model")
        print(f"{config['name']:<10} acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

        # Capture the run ID 
        if acc > best_acc:
            best_acc = acc
            best_run = run.info.run_id

print(f"\nBest run ID: {best_run}  (accuracy={best_acc:.3f})")

# 5. Load best model back tracking storage format
best_model = mlflow.sklearn.load_model(f"runs:/{best_run}/model")
print(f"Reloaded best model: {type(best_model).__name__}")
print(f"Reloaded accuracy  : {accuracy_score(y_te, best_model.predict(x_te)):.3f}")

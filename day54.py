import numpy as np
import xgboost as xgb
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

#1.setup synthetic data
x,y=make_classification(
    n_samples=200,
    n_features=20,
    n_informative=15,
    n_classes=2,
    random_state=42
)

x_tr,x_te,y_tr,y_te=train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y)

#2.Objective function for Optuna
def objective():
    param_space = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 180, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42
    }

# Advanced- Set up Cross-Validation to prevent overfitting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold ,(train_idx,val_idx) in enumerate(cv.split(x_tr,y_tr)):
        x_fold_train, x_fold_val = x_tr[train_idx], x_tr[val_idx]
        y_fold_train, y_fold_val = y_tr[train_idx], y_tr[val_idx]
        model = xgb.XGBClassifier(**param_space)
        
        model.fit(
            x_fold_tr,y_fold_tr,eval_set=[( x_fold_val,y_fold_val)],
            verbose=False
        )
        
         
        # Evaluate validation score
        preds = model.predict(x_fold_val)
        score = accuracy_score(y_fold_val, preds)
        cv_scores.append(score)
        
        # Advanced: Optuna Pruning 
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return np.mean(cv_scores)

if __name__ == "__main__":
     # 4. Execute optimization process
    print("Beginning Bayesian optimization process...")
    study.optimize(objective, n_trials=50, timeout=100)
    
    # 5. Extract and print results
    print("\n--- Optimization Complete ---")
    print(f"Best Trial Score (CV Accuracy): {study.best_value:.4f}")
    print("Optimal Hyperparameters:")
    for key, value in study.best_params_.items():
        print(f"  {key}: {value}")
    
    # 6. Train the final production model 
    print("\nTraining final production model...")
    best_model = xgb.XGBClassifier(**study.best_params_, random_state=42)
    best_model.fit(x_tr, y_tr)
    
    # Test final performance 
    final_preds = best_model.predict(x_te)
    test_accuracy = accuracy_score(y_te, final_preds)
    print(f"Unseen Test Set Accuracy: {test_accuracy:.4f}")

    # 7. Advanced Diagnostics:
    try:
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_param_importances(study).show()
    except Exception as e:
        print("\nSkipping local plot rendering (requires notebook environment).")
#Cross Validation & Hyperparameter Tuning
import numpy as np 
from  sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
                                     )
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#breast cancer data load
breast=load_breast_cancer()
x,y=breast.data,breast.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
sc=StandardScaler()
x_tr=sc.fit_transform(x_tr)
x_te=sc.transform(x_te)

#1.cross validation
print("1.Cross Validation--")
print("-" * 50)

model=RandomForestClassifier(n_estimators=100,random_state=42)
#5-fold CV 
scores=cross_val_score(model,x_tr,y_tr,cv=5,scoring='accuracy')
print(f"Each fold score : {scores.round(3)}")
print(f"Mean accuracy   : {scores.mean():.3f}")
print(f"Standarf deviation   : {scores.std():.3f}")
print(f"(low std = stable model)")

#2.GridSearchCV - try every combination
print("\n 2.GridSearchCV Random Forest")
print("=" * 55)
params_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,5,10],
    'min_samples_split':[2,5],
}

grid=GridSearchCV(
    RandomForestClassifier(random_state=42),
    params_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 , #use all cpu cores
    verbose=1
)

grid.fit(x_tr,y_tr)
print(f"Best parameters  : {grid.best_params_}")
print(f"Best CV accuracy  : {grid.best_score_:.3f}")
print(f"Test accuracy: {accuracy_score(y_te, grid.predict(x_te)):.3f}")

#3.RandomizedSearchCV
print("\n RandomizedSearchCV - SVM")
print("=" * 55)
param_dist = {
    'C': np.logspace(-3, 3, 10), # Search from 0.001 to 1000
    'gamma': np.logspace(-3, 3, 10), 
    'kernel': ['rbf', 'poly', 'sigmoid']
}

rand = RandomizedSearchCV(
    SVC(random_state=42),
    param_dist,
    n_iter=10,       # try 10 random combos
    cv=5,
    scoring='accuracy',
    random_state=42,
    verbose=1
)
rand.fit(x_tr,y_tr)

print(f"Best parameters  : {rand.best_params_}")
print(f"Best CV accuracy  : {rand.best_score_:.3f}")
print(f"Test accuracy: {accuracy_score(y_te, rand.predict(x_te)):.3f}")

#4.compare
print("4.Before vs After Tuning:")

# Default SVM
default = SVC(random_state=42)
default.fit(x_tr, y_tr)
default_acc = accuracy_score(y_te, default.predict(x_te))


# Tuned SVM
tuned_acc = accuracy_score(y_te, rand.predict(x_te))

print(f"{'Model':<25} {'Test Accuracy':>14}")
print("-" * 40)
print(f"{'SVM (default)':<25} {default_acc:>14.3f}")
print(f"{'SVM (tuned)':<25} {tuned_acc:>14.3f}")
print(f"\n Improvement: +{(tuned_acc - default_acc):.3f}")



#KNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

#Iris KNN
print("1.KNN Classifier-Iris")
iris=load_iris()
x,y=iris.data,iris.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
sc=StandardScaler()
x_tr=sc.fit_transform(x_tr)
x_te=sc.transform(x_te)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_tr,y_tr)
y_pred=knn.predict(x_te)
report=classification_report(y_te,y_pred,target_names=iris.target_names)
clean='\n'.join([l for l in report.split('\n') if 'accuracy' not in l])
print(clean)
print(f"Overall Accuracy:{accuracy_score(y_te,y_pred):.3f}")

#2.KNN Breast Cancer
print("2.KNN Classifier-Breast Cancer:")
breast=load_breast_cancer()
x,y=breast.data,breast.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
sc2=StandardScaler()
x_tr=sc2.fit_transform(x_tr)
x_te=sc2.transform(x_te)
knn2=KNeighborsClassifier(n_neighbors=10)
knn2.fit(x_tr,y_tr)
y_pred2=knn2.predict(x_te)
report2=classification_report(y_te,y_pred2,target_names=breast.target_names)
clean='\n'.join([l for l in report.split('\n') if 'accuracy' not in l ])
print(clean)
print(f"Overall Accuracy:{accuracy_score(y_te,y_pred2):.3f}")

#3.Find best K
print("3.Finding best K value--")
tr_scores,te_scores,k_range=[], [], range(1,26,2)
print(f"{'K':<6} {'Train Accuracy':>10} {'Test Accuracy':>10}")
for k in k_range:
    m=KNeighborsClassifier(n_neighbors=k)
    m.fit(x_tr,y_tr)
    tr=accuracy_score(y_tr,m.predict(x_tr))
    te=accuracy_score(y_te,m.predict(x_te))
    tr_scores.append(tr)
    te_scores.append(te)
    print(f"{k:<6} {tr:>10.3f} {te:>10.3f}")
best_k=list(k_range)[tr_scores.index(max(te_scores))]
print(f"\n Best K={best_k} with test accuracy={max(te_scores):.3f}")

#4.distance comparison
print("4.Distance Metrics Comparison--")
metrices=['euclidean','manhattan','chebyshev']
print(f"{'Metric:<15'} {'Accuracy:>10'}")
for m in metrices:
    m1=KNeighborsClassifier(n_neighbors=5,m=m1)
    m1.fit(x_tr,y_tr)
    acc=accuracy_score(y_te,m1.predict(x_te))
    print(f"{m:<15} {acc:>10.3f}")




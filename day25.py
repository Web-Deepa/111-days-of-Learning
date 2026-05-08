#Support Vector Machine(SVM)
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris,load_breast_cancer,make_classification,make_circles
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

#1.Linear SVM-Iris
print("1.SVM Classifier-Iris:")
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#scale
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

svm=SVC(kernel='linear',C=1.0,random_state=42)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
report=classification_report(y_test,y_pred,target_names=iris.target_names)
clean='\n'.join([l for l in report.split('\n') if 'accuracy' not in l])
print(clean)
print(f"Overall Accuracy:{accuracy_score(y_test,y_pred):.3f}")

#RBF(gaussian) kernel SVM for Breast cancer
print("2.RBF SVM -Breast Cancer")
breast=load_breast_cancer()
x,y=breast.data,breast.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
scaler2=StandardScaler()
x_tr=scaler2.fit_transform(x_tr)
x_te=scaler2.transform(x_te)
svm2=SVC(kernel='rbf',gamma='scale',random_state=42)
svm2.fit(x_tr,y_tr)
y_pred2=svm2.predict(x_te)
report2=classification_report(y_te,y_pred2,target_names=breast.target_names)
clean='\n'.join([l for l in report2.split('\n') if 'accuracy' not in l])
print(clean)
print(f"Overall Accuracy:{accuracy_score(y_te,y_pred2):.3f}")

#3.Effect of C parameter
C_values=[0.001,0.01,0.1,1,10,100]
train_scores=[]
test_scores=[]
print(f"{'C value':<12} {'Train Accuracy':>10} {'Test Accuracy':>10}")
print("-" * 30)
for c in C_values:
    m=SVC(kernel='rbf',C=c,gamma='scale',random_state=42)
    m.fit(x_tr,y_tr)
    tr=accuracy_score(y_tr,m.predict(x_tr))
    te=accuracy_score(y_te,m.predict(x_te))
    train_scores.append(tr)
    test_scores.append(te)
    print(f"{c:<12} {tr:>10.3f} {te:>10.3f}")
#4.Kernel Comparison
print("3.Kernel Comparison on Non-linear Data:")
#make circle
x_nl,y_nl=make_circles(n_samples=300,noise=0.1,factor=0.4,random_state=42)
x_trn,x_tes,y_trn,y_tes=train_test_split(x_nl,y_nl,test_size=0.2,random_state=42)
sc=StandardScaler()
x_trn=sc.fit_transform(x_trn)
x_tes=sc.transform(x_tes)

kernels=['linear','rbf','poly']
print(f"{'Kernel':<12} {'Accuracy':>10}")
print("-" * 24)
for k in kernels:
    m=SVC(kernel=k,C=1.0,random_state=42)
    m.fit(x_trn,y_trn)
    acc=accuracy_score(y_tes,m.predict(x_tes))
    print(f"{k:<12} {acc:>10.3f}")






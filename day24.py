#random forest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
#1.Iris
print("1.Random Forest Classifier-Iris")
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf1=RandomForestClassifier(n_estimators=100,random_state=42)
rf1.fit(x_train,y_train)
y_pred1=rf1.predict(x_test)
report1=classification_report(y_test,y_pred1,target_names=iris.target_names)
lines=report1.split('\n')
clean1='\n'.join([l for l in lines if 'accuracy'not in l ])
print("-" * 55)
print(clean1)
print(f"Overall Accuracy:{accuracy_score(y_test,y_pred1):.3f}")

#2.Random Forest of Breast Cancer
print("2.Random Forest Classifier-Breast Cancer")
print("=" * 55)
breast=load_breast_cancer()
x,y=breast.data,breast.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
rf2=RandomForestClassifier(n_estimators=100,random_state=42)
rf2.fit(x_train,y_train)
y_pred2=rf2.predict(x_test)
report2=classification_report(y_test,y_pred2,target_names=breast.target_names)
lines=report2.split('\n')
clean2='\n'.join([l for l in lines if 'accuracy'not in l ])
print("-" * 55)
print(clean2)
print(f"Overall Accuracy:{accuracy_score(y_test,y_pred2):.3f}")

#3.m_estomators effect
print("3.n_estimators Effects :no. of trees ")
train_scores,test_scores,n_trees=[],[],range(1,150,10)
for n in n_trees:
    m=RandomForestClassifier(n_estimators=n,random_state=42)
    m.fit(x_train,y_train)
    train_scores.append(accuracy_score(y_train,m.predict(x_train)))
    test_scores.append(accuracy_score(y_test,m.predict(x_test)))
    print(f" Trees={n:3d}|train={train_scores[-1]:.3f}|test={test_scores[-1]:.3f}")
plt.figure(figsize=(8,4))
plt.plot(n_trees,train_scores,'o',label='Train Accuracy',color='steelblue')
plt.plot(n_trees,test_scores,'s',label='Test Accuracy',color='orange')
plt.title("Random Forest Accurcy by no. of Trees")
plt.xlabel("No. of Trees")
plt.ylabel("Accuracy")
plt.legend()
plt.ylim(0.8,1.02)
plt.savefig("n_trees.png")

plt.show()

#4.Random Forest Regressor
print("Random Tree Regressor")
np.random.seed(42)
x_reg=np.sort(np.random.rand(100,1) * 300,axis=0)
x_reg=np.sort(np.random.rand(80,1)*3000,axis=0)
y_reg=50000 + x_reg.flatten() * 120 +np.random.randn(80) * 20000
x_tr,x_te,y_tr,y_te=train_test_split(x_reg,y_reg,test_size=0.2,random_state=42)
rfr=RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(x_tr,y_tr)
x_plot=np.linspace(0,3000,500).reshape(-1,1)
y_plot=rfr.predict(x_plot)
plt.figure(figsize=(8,4))
plt.scatter(x_reg,y_reg,color="steelblue",alpha=0.5,label='Actual data')
plt.plot(x_plot,y_plot,color="red",linewidth=2,label="RF Prediction")
plt.xlabel("House size(sqft)")
plt.ylabel("Prie($)")
plt.title("Random Forest Regressor of House Price Prediction")
plt.legend()
plt.savefig("ran_house.png")
plt.show()




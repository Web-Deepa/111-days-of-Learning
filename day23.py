#decision tree(classification + Regression)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree,export_text
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

#Iris Dataset-Simple Decision Tree
print("1.Decision Tree Classifier-Iris")
print("=" * 55)
iris=load_iris()

x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(max_depth=4,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
report1=classification_report(y_pred,y_test,target_names=iris.target_names)
print(f"Classification report of Iris\n:{report1}")
print(f" Overall Accuracy:{accuracy_score(y_test,y_pred):.2f}")
print("Tree structure(text):")
print(export_text(model,feature_names=list(iris.feature_names)))
#plot tree
plt.figure(figsize=(14,6))
plot_tree(model,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,rounded=True, fontsize=10)
plt.title("Iris Dataset Decision Tree")
plt.savefig("iris.png")
plt.show()

#2.Breast Cancer Decision Tree
print("Decison Tree Classifier-Breast Cancer")
print("=" * 55)
data=load_breast_cancer()
x,y=data.data,data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model2=DecisionTreeClassifier(max_depth=4,random_state=42)
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)
report2=classification_report(y_pred,y_test,target_names=data.target_names)
print(f"Classification Report of Breast Cancer")
print(report2)
print(f"Overall Accuracy:{accuracy_score(y_test,y_pred):.2f}")
print("Tree Structure of Breast Cancer Classifier:")
print(export_text(model2,feature_names=list(data.feature_names)))
#plot tree
plt.figure(figsize=(14,6))
plot_tree(model2,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,rounded=True,fontsize=8)
                  

plt.savefig("breast.png")
plt.show()

#3.Decison Tree Regressor
print("Decision Tree Regressor")
np.random.seed(42)
x_reg=np.sort(np.random.rand(80,1)*3000,axis=0)
y_reg=50000 + x_reg.flatten() * 120 +np.random.randn(80) * 20000
x_tr,x_te,y_tr,y_te=train_test_split(x_reg,y_reg,test_size=0.2,random_state=42)
reg=DecisionTreeRegressor(max_depth=4,random_state=42)
reg.fit(x_tr,y_tr)
x_plot=np.linspace(0,3000,500).reshape(-1,1)
y_plot=reg.predict(x_plot)
plt.figure(figsize=(8,4))
plt.scatter(x_reg,y_reg,color="steelblue",alpha=0.5,label="Actual Data")
plt.plot(x_plot,y_plot,color="red",linewidth=2 ,label="Tree Prediction")
plt.xlabel("House size(sqft)")
plt.ylabel("Price($)")
plt.title("House Price ")
plt.savefig("house.png")
plt.show()


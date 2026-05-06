#Logistic Regression(Classification)

#1.Simple Logistic Regression(sklearn)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns

#sample:predict if student pass (1) or fail(0) based on study hour
x=np.array([[1],[2],[3],[4],[5]])
y=np.array([1,0,0,1,1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)#prediction
y_prob=model.predict_proba(x_test) #probablity
print("Predictions:",y_pred)
print("Probablities:",y_prob)
print("Accuracy :",accuracy_score(y_test,y_pred))
print("Classification report:")
print(classification_report(y_test,y_pred))

#2.RealDataset(Breat Cancer)
data=load_breast_cancer()
x,y=data.data,data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#important :always scale features for logistic regression
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

report = classification_report(y_test, y_pred, target_names=data.target_names, output_dict=True)
print("Classification Report of Breast Cancer:")
print(report)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}")

#confusion matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=data.target_names,yticklabels=data.target_names)
plt.title("Confusion  Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("breast.png")
plt.show()

#3.scratch implementation
class LogisticRegressionScratch:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,x,y):
        n,feat=x.shape
        self.w=np.zeros(feat)
        self.b=0
        
        for e in range(self.epochs):
            z=np.dot(x,self.w)+self.b
            y_pred=self.sigmoid(z)

            loss=-np.mean(y*np.log(y_pred + 1e-9)+(1-y)*np.log(1-y_pred + 1e-9))
            dw=(1/n) * np.dot(x.T,(y_pred-y))
            db=(1/n) * np.sum(y_pred - y)
            self.w -=self.lr * dw
            self.b -=self.lr *db

            if e % 200 == 0:
                print(f"Epoch{e:4d}|Loss:{loss:.df}")

def predict_Proba(self,x):
    return self.sigmoid(np.dot(x,self.w)+self.b)

def predict(self,x,threshold=0.5):
    return (self.predict_proba(x) >=threshold).astype("Int64")
#test
x,y =make_classification(n=500,feat=5,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegressionScratch(lr=0.1,epochs=1000)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(f"Accuracy:{accuracy_score(y_test,pred):.4f}")


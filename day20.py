#first real ML Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report)

print("Titanic Survival Prediction")
print("=" * 55)

#load data
df=sns.load_dataset("titanic")

#clean data
df["age"]=df["age"].fillna(df["age"].median())
df["embarked"]=df["embarked"].fillna(df["embarked"].mode()[0])
df.drop("deck",axis=1,inplace=True)

#featuring
df["family_size"]=df["sibsp"]+df["parch"]+1 #familysize
df["is_alone"]=(df["family_size"]==1).astype("Int64") #isalone
df["age_group"]=pd.cut(df["age"], #agegroup
                       bins=[0,12,18,35,60,100],
                       labels=[0,1,2,3,4])
df["age_group"]=df["age_group"].cat.codes
df["fare_group"]=pd.cut(df["fare"], #faregroup
                        bins=[0,10,30,100,600],
                        labels=[0,1,2,3])
df["fare_group"]=df["fare_group"].cat.codes

#text to num
df["sex_num"]=df["sex"].map({"male":0,"female":1})
df["embarked_num"]=df["embarked"].map({"S":0,"C":1,"Q":2})
print(f"total passengers:{len(df)}")

#select features
fin_feat=["pclass","sex_num","age_group","fare_group"
          ,"family_size","is_alone","embarked_num"]
x=df[fin_feat].copy()
y=df["survived"].copy()
print(f"Feature:{fin_feat}")
print(f"X shape:{x.shape}")
print(f"Y shape:{y.shape}")

#train test
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=42

)
print(f"training data:{len(x_train)} passengers")
print(f"testing data :{len(x_test)} passengers")
print(f"training survived:{y_train.sum()}" f"({y_train.mean()*100:.2f}%)")
print(f"testing survived:{y_test.sum()}" f"({y_test.mean()*100:.2f}%)")

#ml model
print("Algorith:Logistic Regression\n Best for:yes/no predictions \n question:survive or not?")
model=LogisticRegression(max_iter=1000) #model create
print("model created")

#train model
print("Training the model from data")
model.fit(x_train,y_train) #train
print("training complete")

#predictions
y_pred=model.predict(x_test) #make predictions
print(f"Predictions made for{len(y_pred)}passengers")
print(f"{'#':<5} {'Predicted':>10} {'Actual':>10} {'Correct':>8}")
print("-" * 38)
for i in range(15):
    predicted="Survived" if y_pred[i]==1 else "died"
    actual="Survived" if y_test.iloc[i]==1 else "died"
    correct="✅" if y_pred[i]==y_test.iloc[i] else "❌"
    print(f"{i+1:<5} {predicted:>10} {actual:>10} {correct:>8}")

#evaluate model
accuracy=accuracy_score(y_test,y_pred)  
print(f"Accuracy(AI correctly predicted):{accuracy*100:.2f}%")  

#confusion matrix
print("Confusion matrix--")
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(f"""
Confusion matrix meaning:
      True Died(correctly predicted died):{cm[0][0]}
      False Survived(predicted died ,actually survived):{cm[0][1]}
      False Died(predicted survived ,actually died):{cm[1][0]}
      True Survived(correctly survived as predicted):{cm[1][1]}

""")
#classification report
print("--Detailed report--")
print(classification_report(y_test,y_pred,target_names=["Died","Survived"]))

#feature imp
print("which features did AI use most?")
imp=pd.DataFrame({
    "feature":fin_feat,
    "coff":abs(model.coef_[0])
}).sort_values("coff",ascending=False)
print(f"{'Features':<15} {'Importance':>12}")
print("-" * 50)
for _,r in imp.iterrows():
    bar="█" * int(r["coff"] * 10)
    print(f"{r['feature']:<15} {r['coff']:>8.3f} {bar}")

#predicate new passengers
print("Let predicate survival for new passengers")
new_pasg=pd.DataFrame({
    "pclass":[1,3,2],
    "sex_num":[1,0,1],
    "age_group":[2,1,1],
    "fare_group":[3,0,1],
    "family_size":[2,1,3],
    "is_alone":[0,1,0],
    "embarked_num":[0,0,1]
})
desc=[
    "Rich woman,1st class, with family"
    "Poor man,3rd class,alone"
    "Middle woman,2nd class,with family"
]
pred=model.predict(new_pasg)
prob=model.predict_proba(new_pasg)
print(f"{'Passenger':<40} {'Prediction':>10} {'Probablity':>12}")
print("-" *60)
for d,p,pr in zip(desc,pred,prob):
    res="Survived ✅" if p==1 else "Died ❌"
    surv_prob=pr[1]*100
    print(f"{d:<40} {res:>10} {surv_prob:>10.1f}%")

#visualization chart
fig, axes=plt.subplots(1,2, figsize=(15,5))  
fig.suptitle("First Ml Model Results",fontsize=16)
#confusion chart
im=axes[0].imshow(cm,cmap="Blues")
axes[0].set_title("Confusion Matrix") 
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticks([0,1])
axes[0].set_yticks([0,1])
axes[0].set_xticklabels(["Died","Survived"])
axes[0].set_yticklabels(["Died","Survived"])
for i in range(2):
    for j in range(2):
        axes[0].text(j,i,cm[i,j],
                     ha="center",va="center",
                     fontsize=20,fontweight="bold")

#feature importance
axes[1].barh(imp["feature"],
             imp["coff"],
             color="steelblue")
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Importance score")   
plt.savefig("mlres.png")
plt.show()        

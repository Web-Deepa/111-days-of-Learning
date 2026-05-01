#data cleaning + ML preparation
import numpy as np
import pandas as pd
import seaborn as sns

print("Titanic data cleaning")
df=sns.load_dataset("titanic") #data loading
print(f"original shape:{df.shape}")
print(df.head())

#check problems
print(df.isnull().sum()) #checking null value

#fixing missing values
med_age=df["age"].median() #fill null age with median age
df["age"].fillna(med_age,inplace=True)
print(f"age filled with median:{med_age}")

most_com = df["embarked"].mode()[0] #fix embarked with most common
df["embarked"].fillna(most_com,inplace=True)
print(f" embarked filled with:{most_com}")

df.drop("deck", axis=1,inplace=True) #drop deck cols 
print("missing value after fixing")
print(df.isnull().sum())

#select features
feat=["pclass","sex","age","sibsp","parch","fare","embarked"]
target="survived"
print(f"featured:{feat}")
print(f"target:{target}")
print("(target=what AI will predicate)")

#convert text to number
df["sex_num"]=df["sex"].map({"male":0,"female":1}) #convert sex
print("male ->0")
print("female->1")

df["embarked_num"]=df["embarked"].map({"S":0,"C":1,"Q":2}) #embarked convert
print("S->0")
print("C->1")
print("Q->2")

#final feature matrix
ml_feat=["pclass","sex_num","age","sibsp","parch","fare","embarked_num"]
x=df[ml_feat]
y=df[target]
print(f"X shape(features):{x.shape}")
print(f"Y shaape(target):{y.shape}")
print(x.head())
print(y.head().to_string())

#normalization
x_norm=x.copy()
for col in["age","fare"]:
    min=x_norm[col].min()
    max=x_norm[col].max()
    x_norm[col]=(x_norm[col]-min)/(max-min)
print("first 5 rows after normalization")
print(x_norm.head())    

#train test spilt
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    x_norm,y,test_size=0.2,random_state=42 #20% test,same spilt every time
)

print(f"total data:{len(x_norm)} passengers")
print(f"Training data:{len(x_train)} passengers =80%")
print(f"testing data:{len(x_test)} passengers=20%")

#summary
print(f"Trainig survived:{y_train.sum()}" + f"({y_train.mean()*100:.2f}%)")
print(f"testing survived:{y_test.sum()}"  + f"({y_test.mean()*100:.2f}%)")

#save to csv file
x_train.to_csv("X_train.csv",index=False)
x_test.to_csv("X_test.csv",index=False)
y_train.to_csv("Y_train.csv",index=False)
y_test.to_csv("Y_test.csv",index=False)



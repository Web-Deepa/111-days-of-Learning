#feature engineering
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("loading dataset---")
df=sns.load_dataset("titanic") #load clean data
df["age"].fillna(df["age"].median(),inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0],inplace=True)
df.drop("deck",axis=1,inplace=True)
print(df)
print(f"{df.shape}")

#featuring--
print("creating new features from existing data--")
df["family_size"]=df["sibsp"]+df["parch"]+1 #family size
print(df[["sibsp","parch","family_size"]].head())

#is alone
df["is_alone"]=(df["family_size"]==1).astype(int) 
print("alone:")
print("0=has family")
print(" 1=alone")

 #age group
df["age_group"]=pd.cut(df["age"],bins=[0,12,18,35,60,100],
                       labels=[0,1,2,3,4])
df["age_group"]=df["age_group"].astype("Int64")
print("age group:")
print("Age group--")
print("0=child(0-12)")
print("1=Teen(12-18)")
print("2=Adult(18-35)")
print("3=Middle(35-60)")
print("4=Senior(60+)")

#fare group
df["fare_group"]=pd.cut(df["fare"],bins=[0,10,30,100,600]
                        ,labels=[0,1,2,3])
df["fare_group"]=df["fare_group"].astype("Int64")
print("fare group:")
print("0=cheap(0-10)")
print("1=Normal(10-30)")
print("2=Expensive(30-100)")
print("3=very expensive(100+)")

#text to numbers
df["sex_num"]=df["sex"].map({"male":0,"female":1})
df["embarked_num"]=df["embarked"].map({"S":0,"C":1,"Q":2})
print("sex: male=0,female=1")
print("embarked: S=0,C=1,Q=2")

#feature imp
print("survival rate by gender:")
print(df.groupby("sex")["survived"].mean().round(2))
print("survival rate by class:")
print(df.groupby("pclass")["survived"].mean().round(2))
print("survival rate by alone/family:")
print(df.groupby("is_alone")["survived"].mean().round(2))
print("survival rate by age group:")
age_lbl={0:"Child",1:"Teen",2:" Adult",3:"Middle",4:"Senior"}
age_surv=df.groupby("age_group")["survived"].mean()
for g,r in age_surv.items():
    print(f"{age_lbl[g]:<8}:{r:.2%} survived")
print("survival rate by fare group:")    
f_lbl={0:"Cheap",1:"Normal",2:"Expensive",3:"Very Expensive"}
f_surv=df.groupby("fare_group")["survived"].mean()
for g,r in f_surv.items():
    print(f"{f_lbl[g]:<15}:{r:.2%} survived")

#correlation
n_feat=["pclass","sex_num","age","fare","family_size","is_alone","embarked_num"]  
corrl=df[n_feat+["survived"]].corr()["survived"]  
corrl=corrl.drop("survived").sort_values(ascending=False)
print(f"{'Features':<15} {'Correlation':>12} {'Meaning':>10}")
print("-" *40)
for f,c in corrl.items():
    if c>0:
        meaning="helps survive"
    else:
        meaning="reduces survival"    
    print(f"{f:<15} {c:>12.2f} {meaning:>20}")    

#final feature set
fin_feat=["pclass","sex_num","age_group","fare_group","family_size","is_alone","embarked_num"]
x=df[fin_feat].copy()
x["age_group"].astype(str).astype("Int64")
x["fare_group"].astype(str).astype("Int64")
y=df["survived"].copy()
print(f"Features selected:{fin_feat}")
print(f"X shape:{x.shape}")    
print(f"Y-shape:{y.shape}")
print("firsst 5 rows:\n",x.head())

#chart visualization
fig,axes=plt.subplots(2,2,figsize=(12,16))
fig.suptitle("Features Analysis",fontsize=18)

#family size
f_surv=df.groupby("family_size")["survived"].mean() * 100
axes[0,0].bar(f_surv.index,f_surv.values,color="blue")
axes[0,0].set_title("Survival by Family size")
axes[0,0].set_xlabel("Family size")
axes[0,0].set_ylabel("Survival rate %")


#alone
a_surv=df.groupby("is_alone")["survived"].mean() *100
axes[0,1].bar(["Has family","Alone"], a_surv.values,color=["green","red"])
axes[0,1].set_title("Survival by alone/family ")
axes[0,1].set_ylabel("survival rate %")

#age group
ag_surv=df.groupby("age_group")["survived"].mean() * 100
axes[1,0].bar(["Child","Teen","Adult","Middle","Senior"],ag_surv.values,color="purple")
axes[1,0].set_title("Survival by age group")
axes[1,0].set_ylabel("survival rate %")

#fare group
f_surv=df.groupby("fare_group")["survived"].mean() *100
axes[1,1].bar(["cheap","Normal","Expensive","Very Expensive"],f_surv.values,color="orange")
axes[1,1].set_title("Survival by fare group")
axes[1,1].set_ylabel("Survival rate %")

plt.savefig("features.png")
plt.show()

#save
x.to_csv("X_final.csv" ,index=False)
y.to_csv("Y_final.csv" ,index=False)
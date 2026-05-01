#titanic dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Titanic dataset exploration")
print("=" * 55)

#1.load dataset
print("loading titanic dataset--")
df=sns.load_dataset("titanic")
print("data laoded successfully")
print(f"Dataset Shape:{df.shape}")
print(f"Rows:{df.shape[0]} passengers")
print(f"Columns:{df.shape[1] } features")

#2.first look
print("First 5 rows--")
print(df.head())
print("Column name")
print(df.columns.tolist())
print("Data types")
print(df.dtypes)

#3.understanding columns
print("Columns meaning")
mean={
    "survived": "0 = died, 1=survived" ,
    "pclass":"Ticket class(1=1st,2=2nd,3=3rd)" ,
    "sex":"male or female" ,
    "age":"age of passengers",
    "sibsp":"siblings/spouses on board" ,
    "parch": "parents or children on board" ,
      "fare":"ticket  price paid" ,
      "embarked":"port where boarded(C/Q/S)"
      }
for c,m in mean.items():
    print(f"{c:<12}:{m}")

#4.basic statistics:
print("basic statistics--")
print(df.describe())

#5.missing values
print("missing values--")
miss=df.isnull().sum()
miss_pct=(df.isnull().sum()/len(df) * 100).round(1)
print(f"{'column':<15} {'missing':>5} {'percent':>10}")
print("-" * 40)
for c in df.columns:
    if miss[c] > 0:
        print(f"{c:<15} {miss[c]:>6} {miss_pct[c]:>7}%")

#6.survival analysis
print("survival analysis--")
total=len(df)
survived=df["survived"].sum()
died=total-survived
surv_rate=survived/total * 100
print(f"total passemgers:{total}")
print(f"survived:{survived} ({surv_rate:.1f}%)")
print(f"died: {died} ({100-surv_rate:.1f}%)")

#by gender
print("survival by gender--")
g_surv=df.groupby("sex")["survived"].mean() *100
for g,r in g_surv.items():
    print(f"{g:<8}: {r:.1f}% survived")

#by class
print("survival by ticket class--")
c_surv=df.groupby("pclass")["survived"].mean() *100
for pclass,r in c_surv.items():
    label={1:"First",2:"Second",3:"Third"}[pclass]
    print(f"{label:<8}: {r:.2f}% survived")  

#by  age group      
print("survival by age--")
df["age_grp"]=pd.cut(df["age"],
                    bins=[0,12,18,35,60,100],
                     labels=["Child" ,"Teen" ,"Adult","Middle","Senior"] )
a_surv=df.groupby("age_grp")["survived"].mean() * 100
for a,r in a_surv.items():
    print(f"{str(a):<8}: {r:.2f}% survived")

#7.visualization
print("Creating visualization--")
fig,axes=plt.subplots(2,2, figsize=(12,8))
fig.suptitle("Titanic Data Analysis",fontsize=16)

#chart:survival count
axes[0,0].bar(["died,survived"],[died,survived], color=["red","green"])
axes[0,0].set_title("Survival count")
axes[0,0].set_ylabel("No. of passengers")

#survial by gender
gen_data=df.groupby("sex")["survived"].mean() * 100
axes[0,1].bar(gen_data.index,gen_data.values,color=["lightblue","pink"])
axes[0,1].set_title("Survival rate by gender")
axes[0,1].set_ylabel("survival rate %")

#survival by class
pclass_data=df.groupby("pclass")["survived"].mean() * 100
axes[1,0].bar(pclass_data.index,pclass_data.values,color=["gold","silver","brown"])
axes[1,0].set_title("Survival rate by class")
axes[1,0].set_ylabel("survival rate %")

#age distribution
axes[1,1].hist(df["age"].dropna(),bins=20,color="purple",edgecolor="white")
axes[1,1].set_title("Age Distribution")
axes[1,1].set_xlabel("Age")
axes[1,1].set_ylabel("Count")
plt.savefig("titanic_analysis.png")
df.to_csv("titanic_explored.csv",index=False)

plt.show()



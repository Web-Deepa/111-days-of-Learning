#  Day 50 — End to End ML Project: Titanic (Short)
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


#1. Data load
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df  = pd.read_csv(url)
print(f"Shape: {df.shape}")
print(f"Missing:\n{df.isnull().sum()[df.isnull().sum()>0]}")


#2. Feature Engineering
def engineer(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr',
         'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']   = (df['FamilySize'] == 1).astype(int)
    df['HasCabin']  = df['Cabin'].notna().astype(int)
    return df

df = engineer(df)

features  = ['Pclass','Sex','Age','Fare','Embarked',
             'Title','FamilySize','IsAlone','HasCabin']
num_cols  = ['Age','Fare','FamilySize']
cat_cols  = ['Pclass','Sex','Embarked','Title','IsAlone','HasCabin']

x = df[features]
y = df['Survived']
x_tr, x_te, y_tr, y_te = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)


# 3.Pipelining
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler())]), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('enc', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

pipe = Pipeline([
    ('pre', pre),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
])


# 4.Train and Evaluate
pipe.fit(x_tr, y_tr)
y_pred = pipe.predict(x_te)
y_prob = pipe.predict_proba(x_te)[:, 1]

cv = cross_val_score(pipe, x_tr, y_tr, cv=5)
print(f"\n CV Mean   : {cv.mean():.3f}  Standard Deviation: {cv.std():.3f}")
print(f"Test Accuracy  : {accuracy_score(y_te, y_pred):.3f}")
print(f"ROC-Accuracy  : {roc_auc_score(y_te, y_prob):.3f}")

report = classification_report(y_te, y_pred,
         target_names=['Not Survived','Survived'])
print('\n'.join([l for l in report.split('\n') if 'accuracy' not in l]))


# 5.Save Model
joblib.dump(pipe, "titanic_model.pkl")
print("Model Saved")


# 6.Predict New passengers
new = pd.DataFrame({
    'Pclass':['1','3','2'], 'Sex':['female','male','female'],
    'Age':[28,22,35],       'Fare':[100,7.5,25],
    'Embarked':['S','S','C'],'Title':['Mrs','Mr','Miss'],
    'FamilySize':[2,1,1],    'IsAlone':[0,1,1],
    'HasCabin':[1,0,0]
})
preds = pipe.predict(new)
probs = pipe.predict_proba(new)[:,1]
print("\nNew passenger predictions:")
for i,(pred,prob) in enumerate(zip(preds,probs)):
    print(f"  Passenger {i+1}: {'Survived🟢' if pred else 'Not survived 🔴'}  ({prob:.3f})")
#pipelines and preproccessing
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from  sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
                                                                    
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score

#Basic pipeline
print("Basic pipeline")
print("=" * 55)
bst=load_breast_cancer()
x,y=bst.data,bst.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
pipe=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LogisticRegression(max_iter=1000))
])
pipe.fit(x_tr,y_tr)
y_pred=pipe.predict(x_te)
print(f"Accuracy:{accuracy_score(y_te,y_pred):.3f}")
print(f"Steps:{[s[0] for s in pipe.steps]}")

#pipeline with mixed data
print("\n ColumnTransformer - mixed data")
print("=" * 55)
df=pd.DataFrame ({
    'size':[1000,1500,np.nan,2000,2500,2200],
    'bedrooms':[2,3,4,5,np.nan,4],
    'location':['rural','urban','urban','rural','rural','urban'],
    'condition':['good','bad','bad','good','good','good'],
    'sold':[1,1,0,0,1,0]
})
x=df.drop('sold',axis=1)
y=df['sold']
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
num_cols=['size','bedrooms']
cat_cols=['location','condition']

#fill missing value
num_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='median')), 
    ('scaler',StandardScaler())
])
cat_pipe=Pipeline([
     ('imputer',SimpleImputer(strategy='most_frequent')), 
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

#combine both
preprocessor=ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])

#full pipeline
full_pipe=Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestClassifier(n_estimators=100,random_state=42))
])
full_pipe.fit(x_tr,y_tr)
y_pred2=full_pipe.predict(x_te)
print(f"Accuracy:{accuracy_score(y_te,y_pred2):.3f}")

#pipeline + cross validation
print("\n Pipeline and Cross Validation--")
iris=load_iris()
x,y=iris.data,iris.target
pipe_cv=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LogisticRegression(max_iter=1000))
])
scores = cross_val_score(pipe_cv, x, y, cv=5, scoring='accuracy')
print(f"CV scores : {scores.round(3)}")
print(f"Mean      : {scores.mean():.3f}")
print(f"Stadard Deviation      : {scores.std():.3f}")

#Pipeline and Gridsearch
bst=load_breast_cancer()
x,y=bst.data,bst.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
pipe_gs=Pipeline([
    ('scaler',StandardScaler()),
    ('model',LogisticRegression(max_iter=1000))
])
param_grid={
    'model__C':[0.1,1,10],
    'model__solver':['lbfgs','liblinear']
}
grid=GridSearchCV( pipe_gs , param_grid,cv=5, scoring='accuracy',n_jobs=-1 )
grid.fit(x_tr,y_tr) 
print(f"Best parameters  : {grid.best_params_}")
print(f"Best CV accuracy  : {grid.best_score_:.3f}")
print(f"Test accuracy: {accuracy_score(y_te, grid.predict(x_te)):.3f}")









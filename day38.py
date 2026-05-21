#ML Project Workflow
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
 
#1.data collection and splitting
np.random.seed(42)
data=pd.DataFrame({
    'size':np.random.randint(800,3500,size=100),
    'bedrooms':np.random.randint(1,5,size=100),
    'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], size=100),
    'price': np.random.randint(150000, 600000, size=100)
    }) 
print(data)
#sepearte targets and features
x=data.drop(columns=['price'])
y=data['price']

# Separate targets and features
x = data.drop(columns=['price'])
y = data['price']

# Split data 
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)

#2.feature engineering
num_feat=['size','bedrooms']
cate_feat=['neighborhood']

#processing data
num_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                ('scaler',StandardScaler())])

cate_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                ('onehot',OneHotEncoder(handle_unknown='ignore'))
   
])

#Bundle 
preproc=ColumnTransformer(transformers=[
    ('num',num_transformer,num_feat),
    ('cat',cate_transformer,cate_feat)])

#3.MODEL configuration and training
full_pipeline=Pipeline(steps=[
    ('preprocessor',preproc),
    ('regressor',RandomForestRegressor(n_estimators=100,random_state=42))
])

#train entire system
full_pipeline.fit(x_tr,y_tr)

#4.Model Evaluation
y_pred=full_pipeline.predict(x_te)

#calculate validation scores
rmse=np.sqrt(mean_squared_error(y_te,y_pred))
r2=r2_score(y_te,y_pred)
print(f"\n -- Model Performance --")
print(f"Root Mean Squared Error(RMSE):${rmse:,.2f}")
print(f"R-squared (R²) Score:{r2:.4f}")

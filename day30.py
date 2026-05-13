#Feature Engineering
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.DataFrame ({
    'size':[1000,1500,np.nan,2000,2500,2200],
    'bedrooms':[2,3,4,5,np.nan,4],
    'bathrooms':[1,2,2,2,1,1],
    'house_age':[2,4,6,8,5,7],
    'location':['rural','suburb','urban','rural','rural','suburb'],

    'sold':[1,1,0,0,1,0]
})

print("Original data:")
print(df.head())
#creation
print("1.Feature Creation---")
#new features from existing ones
df['size_per_bedroom']=df['size']/df['bedrooms']
df['total_rooms']        = df['bedrooms'] + df['bathrooms']   # room count
df['is_new']             = (df['house_age'] < 5).astype(int)  # binary flag
df['age_squared']        = df['house_age'] ** 2 
print(df[['size_per_bedroom','total_rooms','is_new','age_squared']].head()) 

#transformation
print("\n Transformation--")
df['log_size']=np.log1p(df['size'])
df['sqrt_size']=np.sqrt(df['size'])
print(f"Original size range:{df['size'].min()} -{df['size'].max()}")
print(f"Log size range:{df['log_size'].min():.2f}-{df['log_size'].max():.2f}")
print(f"Sqrt size range:{df['sqrt_size'].min():.1f}-{df['sqrt_size'].max():.1f}")

#3.Encoding
print("\n 3.Encoding Features--")
df_ohe=pd.get_dummies(df,columns=['location'],prefix='loc')
print("After one-hot encoding location:")
print([c for c in df_ohe.columns if 'loc' in c])
lbl=LabelEncoder()
df['location_encoded']=lbl.fit_transform(df['location'])
print(f"\n Label encoded:{dict(zip(lbl.classes_,lbl.transform(lbl.classes_)))} ")

#interaction 
print(" \n 4.Interaction Features--")
df['size_x_rooms']=df['size'] * df['total_rooms']
df['bed_bath_ratio']=df['bedrooms']/df['bathrooms']
print("Interaction Features--")
print(df[['size_x_rooms','bed_bath_ratio']].head())

#binning
print("\n 5.Binning--")
df['age_group']=pd.cut(
    df['house_age'],
    bins=[0,5,15,50],
    labels=['new','mid','old']

)

df['size_category']=pd.qcut(
    df['size'],
    q=3,
    labels=['small','medium','large']

)
print(df[['house_age','age_group','size','size_category']])

#compare
print("\n 6.Before vs After Comparison--")
y=df['sold']

#before
x_before=df[['size','bedrooms','bathrooms','house_age']]
x_tr,x_te,y_tr,y_te=train_test_split(x_before,y,test_size=0.2,random_state=42)
m1=RandomForestClassifier(n_estimators=100,random_state=42)
m1.fit(x_tr,y_tr)
acc_before=accuracy_score(y_te,m1.predict(x_te))

#after
x_after=df[[
    'size','bedrooms','bathrooms','house_age',
    'size_per_bedroom','total_rooms','is_new',
    'log_size','size_x_rooms','bed_bath_ratio',
    'location_encoded'
]]
x_tr2,x_te2,y_tr2,y_te2=train_test_split(x_after,y,test_size=0.2,random_state=42)
m2=RandomForestClassifier(n_estimators=100,random_state=42)
m2.fit(x_tr2,y_tr2)
acc_after=accuracy_score(y_te2,m2.predict(x_te2))

print(f"{'Model':<30} {'Accuracy':>10}")
print("-" * 42)
print(f"{'Before (4 features)':<30} {acc_before:>10.3f}")
print(f"{'After  (11 features)':<30} {acc_after:>10.3f}")
print(f"\n Improvement: +{acc_after - acc_before:.3f}")

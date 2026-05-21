#Model Saving
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

#save trained model
def save_model(model_pipeline:Pipeline,filename:str="trained_model.joblib"):
    joblib.dump(model_pipeline,filename) #serialize and save 

#load and  use model

def load_predict(filename:str,raw_input_data:pd.DataFrame):
    loaded_pipeline=joblib.load(filename) 
    print(f"Model pipeline loaded from:{filename}")
    pred=loaded_pipeline.predict(raw_input_data) #predictions
    return pred

#create to mock data and pipeline
x_tr,y_tr=make_regression(n_samples=10,n_features=2,random_state=42)
df_tr=pd.DataFrame(x_tr,columns=['feature1','feature2'])

#train pipeline
mock_pipeline=Pipeline([('regressor',LinearRegression())])
mock_pipeline.fit(df_tr,y_tr)

#create input data
df_new_input=pd.DataFrame([[0.5,-0.2]],columns=['feature1','feature2'])

#save model
save_model(mock_pipeline,"train_model.joblib")
print(df_new_input)

preds=load_predict("train_model.joblib",df_new_input)
print("Predictions:",preds)
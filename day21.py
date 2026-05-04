
#1.Simple linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

#sample data:
x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y=np.array([12000,14000,16000,18000,20000,22000,24000,26000,28000,30000])
print(x)
print(y)

#spilt data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#train model
model=LinearRegression()
model.fit(x_train,y_train)

#predict
y_pred=model.predict(x_test)

#evaluate
print(f"Weight(slope):{model.coef_[0]:.2f}")
print(f"Bias(intercept):{model.intercept_:.2f}")
print(f"R² Score:{r2_score(y_test,y_pred):.4f}")
print(f"RMSE:{np.sqrt(mean_squared_error(y_test,y_pred)):.2f}")

#plot
plt.scatter(x,y,color="blue", label="Actual")
plt.plot(x,model.predict(x),color="red",label="Predicted Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.savefig("sim_reg.png")
plt.legend()
plt.show()


#2.multiple regression line
data={
    'area':[1000,1500,1800,2000,2500],
    "bedrooms":[2,3,4,5,6],
    "age":[10,5,8,3,4],
    "price":[20000,30000,40000,50000,60000]

}
df=pd.DataFrame(data)
x=df[['area','bedrooms','age']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Cofficients:")
for feat,coef in zip(x.columns,model.coef_):
    print(f"{feat}:{coef:.2f}")
print(f"Intercept:{model.intercept_:.2f}")   
print(f"R² Score:{r2_score(y_test,y_pred):.4f}") 
print(f"RMSE:{np.sqrt(mean_squared_error(y_test,y_pred)):.2f}")

#predicate a new house
new_house=pd.DataFrame([[2000,3,4]],columns=['area','bedrooms','age'])
print(f"Predicted price for new house:${model.predict(new_house)[0]:,.0f}")

#3.scratch implementation
class LinearRegressionScratch:
    def __init__(self,learning_rate=0.01,epochs=1000):
        self.lr=learning_rate
        self.epochs=epochs
        self.w=None
        self.b=None
    def fit(self,x,y):
        n=len(x)
        self.w=0.0 
        self.b=0.0 
        for e in range(self.epochs):
            y_pred=self.w * x + self.b
            loss = np.mean((y_pred-y)** 2) #MSE
            #gradient
            dw=(2/n) * np.sum((y_pred-y) * x)
            db=(2/n) * np.sum(y_pred-y)
            #update weight
            self.w -=self.lr * dw 
            self.b -=self.lr * db
            if e % 100 ==0:
                print(f"Epoch{e:4d}|Loss:{loss:.4f}")
    def predict(self,x):
        return self.w * x + self.b   

#test it
x=np.array([1,2,3,4,5,6,7,8,9,10],dtype=float)
y=np.array([12000,14000,16000,18000,20000,22000,24000,26000,28000,30000])
model=LinearRegressionScratch(learning_rate=0.0001,epochs=1000)
model.fit(x,y)
print(f"Final weight:{model.w:.2f}")
print(f"Final bias:{model.b:.2f}")
print(f"Predict(5yr):{model.predict(5):.0f}")
         


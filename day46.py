#Neural Networks with Keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.metrics import accuracy_score,classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#1.Iris
print("Multi-class classification-Iris---")
iris=load_iris()
x,y=iris.data,iris.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
scal=StandardScaler()
x_tr=scal.fit_transform(x_tr)
x_te=scal.transform(x_te)
model=keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32,activation='relu'),
    layers.Dense(16,activation='relu'),
    layers.Dense(3,activation='softmax')

])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history=model.fit(
    x_tr,y_tr,epochs=80,batch_size=16,validation_split=0.2,verbose=0
)

loss,acc=model.evaluate(x_te,y_te,verbose=0)
print(f"Test Accuracy:{acc:.3f} ")
y_pred=np.argmax(model.predict(x_te,verbose=0),axis=1)
report=classification_report(y_te,y_pred,target_names=iris.target_names)
print('\n'.join([l for l in report.split('\n') if 'accuracy' not in l]))

#2.Dropout
print("Dropout - Preventing Overfitting---")
model2=keras.Sequential([
    layers.Input(shape=(x_tr.shape[1],)),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3,activation='softmax')

])

model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
history2=model2.fit(x_tr,y_tr,epochs=50,batch_size=32,validation_split=0.2,verbose=0)           
loss2,acc2=model2.evaluate(x_te,y_te,verbose=0)
print(f"With Dropout-Test Accuracy:{acc2:.3f}")

#3.Batch Normalization
print("Batch Normalization----")
model3=keras.Sequential([
    layers.Input(shape=(x_tr.shape[1],)),
    layers.Dense(64,activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(32,activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(3,activation='softmax')
   
])

model3.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
history3=model3.fit(x_tr,y_tr,epochs=50,batch_size=32,validation_split=0.2,verbose=0)           
loss3,acc3=model3.evaluate(x_te,y_te,verbose=0)
print(f"With BatchNorm Accuracy:{acc3:.3f}")

#4.Comparison

print(" Model Comparison---")
print(f"{'Model':<30} {'Test Accuracy':>14}")
print("-" * 46)
print(f"{'Basic NN':<30} {acc:>14.4f}")
print(f"{'NN + Dropout':<30} {acc2:>14.4f}")
print(f"{'NN + BatchNorm':<30} {acc3:>14.4f}")






#Transfer Learning
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#1.Preparing Datasets
(img_tr,lbl_tr),(img_te,lbl_te)=mnist.load_data()
img_tr=np.stack([img_tr]*3, axis=-1)/255.0
img_te=np.stack([img_te]*3, axis=-1)/255.0

img_tr=tf.image.resize(tf.convert_to_tensor(img_tr),[32,32])
img_te=tf.image.resize(tf.convert_to_tensor(img_te),[32,32])
lbl_tr=to_categorical(lbl_tr,10)
lbl_te=to_categorical(lbl_te,10)

#2.Building Model
base_model=MobileNetV2(weights='imagenet',include_top=False,input_shape=(32,32,3))
base_model.trainable=False #freeze model

inputs=Input(shape=(32,32,3))
x=base_model(inputs,training=False)
x=GlobalAveragePooling2D()(x)
outputs=Dense(10,activation='softmax')(x)
model=Model(inputs,outputs)

#3.Compiling and Training Model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(img_tr,lbl_tr,epochs=10,validation_split=0.2)

#4.Fine-Tunning Model
base_model.trainable=True
for layer in base_model.layers[:100]:
    layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(img_tr,lbl_tr,epochs=10,validation_split=0.2)

#5.Model Evaluation
loss,accuracy=model.evaluate(img_te,lbl_te)
print(f"Test Loss:{loss:.3f}")
print(f"Test Accuracy:{accuracy:.3f}")

#6.Visualizing Model Performance
te_pred=model.predict(img_te)
te_pred_classes=np.argmax(te_pred,axis=1)
te_true_classes=np.argmax(lbl_te,axis=1)
cm=confusion_matrix(te_true_classes,te_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False)
plt.xlabel("Predicted Lebel")
plt.ylabel("True Label")
plt.title('Confusion Matrix')
plt.savefig("transfer.png")
plt.show()

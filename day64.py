#day64-Image Segmentation(U-Net)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Model

#1.Build a small U-Net
def build_net(input_shape=(128,128,1)):
    inputs=layers.Input(input_shape)
    #Encoder -downsample,extract features
    c1=layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    p1=layers.MaxPooling2D()(c1)
    c2=layers.Conv2D(32,3,activation='relu',padding='same')(p1)
    p2=layers.MaxPooling2D()(c2)

    #bottleneck
    b=layers.Conv2D(64,3,activation='relu',padding='same')(p2)

    #decoder-upsample,combine with encoder features
    u1=layers.UpSampling2D()(b)
    u1=layers.Concatenate()([u1,c2]) #skip connection
    c3=layers.Conv2D(32,3,activation='relu',padding='same')(u1)

    u2=layers.UpSampling2D()(c3)
    u2=layers.Concatenate()([u2,c1]) #skip connection
    c4=layers.Conv2D(16,3,activation='relu',padding='same')(u2)

    outputs=layers.Conv2D(1,1,activation='sigmoid')(c4)
    return Model(inputs,outputs)

model=build_net()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

#2.Fake data
np.random.seed(42)
n=100
x=np.random.rand(n,128,128,1).astype('float32')
y=np.zeros((n,128,128,1),dtype='float32')

for i in range(n):
    cx,cy,r=np.random.randint(40,90,3)
    yy,xx=np.orgid[:128,:128]
    mask=(xx-cx)**2 +(yy-cy)**2 <=r**2 
    y[i,...,0]=mask

#3.Train
model.fit(x,y,epochs=10,batch_size=16,validation_split=0.2,verbose=1)

#4.Predictions
pred=model.predict(x[:1])[0,...,0]
print("Predictions:",pred)
    

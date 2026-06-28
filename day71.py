#Day-71:Autoencoders
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Model

#1.Load MNIST digit images,normalize pixel values to 0-1
(x_tr,_),(x_te,_)=mnist.load_data()
x_tr=x_tr.astype('float32')/255.0
x_te=x_te.astype('float32')/255.0

#Flatten 28x28 images into a single 784 length vector
x_tr_flat=x_tr.reshape(-1,784)
x_te_flat=x_te.reshape(-1,784)

#2.Build Encoder(784 to 32)
input_img=layers.Input(shape=(784,))
encoded=layers.Dense(128,activation='relu')(input_img)
encoded=layers.Dense(32,activation='relu')(encoded) #bottleneck

#3.Build Decoder(32-784)
decoded=layers.Dense(128,activation='relu')(encoded)
decoded=layers.Dense(784,activation='sigmoid')(decoded)

#4.Comnine encoder and deccoder at one
autoencoder=Model(input_img,decoded)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

#5.Train:input and target are same data
autoencoder.fit(
    x_tr_flat,x_tr_flat,
    epochs=10,batch_size=256,
    validation_data=(x_te_flat,x_te_flat),
    verbose=1
)

#6.Build a seperate encoder-only moddel to inspect the compressed representation
encoder_model=Model(input_img,encoded)
compressed=encoder_model.predict(x_te_flat[:5])
print("Original size:784 per image")
print("Compressed size:",compressed.shape[1],"numbers per image")

#7.Reconstruct images and compare visually
reconstructed=autoencoder.predict(x_te_flat[:5])
plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(x_te[i],cmap='gray')
    plt.title("Original:");plt.axis('off')

    plt.subplot(2,5,i+6)
    plt.imshow(reconstructed[i].reshape(28,28),cmap='gray')
    plt.title("Reconstructed");plt.axis('off')
plt.tight_layout()
plt.savefig("autoencoder.png")
plt.show()    
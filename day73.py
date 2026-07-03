#Day73-Generative Adversarial Network(GAN)Basics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Model

#1.Load data and normalize
(x_tr,_),(_,_)=mnist.load_data()
x_tr=(x_tr.astype('float32')-127.5)/127.5
x_tr=x_tr.reshape(-1,784)
latent_dim=100 #size of random noise

#2.Build  generator:random noise->fake 784-pixel image
def build_generator():
    model=tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128,activation='relu'),
        layers.Dense(784,activation='tanh')
    ])
    return model
#3.Build Discriminator:image->real or fake
def build_discriminator():
    model=tf.keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128,activation='relu'),
        layers.Dense(1,activation='sigmoid') #1 real ,0=fake

    ])
    return model
generator=build_generator()
discriminator=build_discriminator()
discriminator.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#4.Combined GAN model:trains only the generator using discriminator's feedback
discriminator.trainable=False
gan_input=layers.Input(shape=(latent_dim,))
fake_img=generator(gan_input)
gan_output=discriminator(fake_img)
gan=Model(gan_input,gan_output)
gan.compile(optimizer='adam',loss='binary_crossentropy')

#5.Training Loop
batch_size=64
epochs=1000
for epoch in range(epochs):
    #train discriminator
    real_imgs=x_tr[np.random.randint(0,x_tr.shape[0],batch_size)]
    noise=np.random.normal(0,1,(batch_size,latent_dim))
    fake_imgs=generator.predict(noise,verbose=0)
    
    d_loss_real=discriminator.train_on_batch(real_imgs,np.ones((batch_size,1)))
    d_loss_fake=discriminator.train_on_batch(fake_imgs,np.zeros((batch_size,1)))

    #Train Generator
    noise=np.random.normal(0,1,(batch_size,latent_dim))
    g_loss=gan.train_on_batch(noise,np.ones((batch_size,1)))

    if epoch %500==0:
        print(f"Epoch:{epochs} \n D_loss:{(d_loss_real[0]+d_loss_fake[0])/2:.3f } \n G_loss:{g_loss:.3f}")

#6.Generate and view new fake digits
noise=np.random.normal(0,1,(5,latent_dim))
gen_imgs=generator.predict(noise,verbose=0)
gen_imgs=(gen_imgs+1)/2

plt.figure(figsize=(10,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(gen_imgs[i].reshape(28,28),cmap='gray')
    plt.title("Fake")
    plt.axis('off')
plt.tight_layout()
plt.savefig("gan.png")     
plt.show()       
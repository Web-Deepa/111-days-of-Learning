import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Model

#1.Load data
(x_tr,_),(x_te,_)=mnist.load_data()
x_tr=x_tr.astype('float32')/255.0
x_te=x_te.astype('float32')/255.0
x_tr_flat=x_tr.reshape(-1,784)
x_te_flat=x_te.reshape(-1,784)
latent_dim=2

#2.Encoder outputs two things instead of one code
inputs=layers.Input(shape=(784,))
h=layers.Dense(128,activation='relu')(inputs)
z_mean=layers.Dense(latent_dim)(h) #center
z_logvar=layers.Dense(latent_dim)(h) #spread

#3.Reparameterization trick
def sampling(args):
    z_mean,z_logvar=args
    epsilon=tf.random.normal(shape=tf.shape(z_mean)) #random noise
    return z_mean+tf.exp(0.5 *z_logvar)*epsilon
z=layers.Lambda(sampling)([z_mean,z_logvar]) #z=latent code

#4.Decoder:rebuild image from latent code
decoder_h=layers.Dense(128,activation='relu')
decoder_out=layers.Dense(784,activation='sigmoid')
decoded_h=decoder_h(z)
outputs=decoder_out(decoded_h)

#5.Custom VAE Loss
class VAE(Model):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder

    def train_step(self,data):
        x=data
        with tf.GradientTape()as tape:
            z_mean,z_logvar,z=self.encoder(x)
            reconstruction=self.decoder(z)
            recon_loss=tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x,reconstruction)
            ) *784
            kl_loss=-0.5*tf.reduce_mean(
             1+z_logvar-tf.square(z_mean)-tf.exp(z_logvar)   
            )   
            total_loss=recon_loss+kl_loss
        grads=tape.gradient(total_loss,self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        return{"loss":total_loss,"recon_loss":recon_loss,"kl_loss":kl_loss}
    
vae=VAE(encoder,decoder)
vae.compile(optimizer='adam')
vae.fit(x_tr_flat,epochs=10,batch_size=256,verbose=1)

#6.Generate brand new digits by Sampling random points from latent space
random_points=np.random.normal(size=(5,latent_dim))
generated=decoder.predict(random_points)

#7.Visualizing
plt.figure(figsize=(10,3))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(generated[i].reshape(28,28),cmap='gray')
    plt.title("Generated image:")
    plt.axis('off')
plt.tight_layout()
plt.savefig('vae.png')
plt.show()    




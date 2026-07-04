# Day-75:Diffusion model 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
# 1: Forward diffusion (adding noise step by step) 
(x_tr, _), (_, _) = mnist.load_data()
sample = x_tr[0].astype('float32') / 255.0   

def add_noise(image, t, T=10): # t = current timestep, T = total timesteps
    noise_level = t / T
    noise       = np.random.normal(0, 1, image.shape)
    return np.clip(image*(1-noise_level) + noise*noise_level, 0, 1)

# 2.Visualize forward process 
plt.figure(figsize=(12,3))
for i, t in enumerate([0, 2, 4, 6, 8, 10]):
    noisy = add_noise(sample, t, T=10)
    plt.subplot(1,6,i+1)
    plt.imshow(noisy, cmap='gray')
    plt.title(f"t={t}"); plt.axis('off')
plt.suptitle("Forward Diffusion: Clean → Noise")
plt.tight_layout()
plt.savefig("day75_forward.png"); plt.show()

#3.: Use pretrained Stable Diffusion via HuggingFace 
print("\nTo run Stable Diffusion (needs GPU + 4GB RAM):")
pretrained_code = '''
from diffusers import StableDiffusionPipeline
import torch

# Load pretrained pipeline (downloads ~4GB first time)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")   # needs GPU

# Generate image from text prompt
image = pipe("a futuristic city at sunset, digital art").images[0]
image.save("generated.png")
'''
print(pretrained_code)
print("Install: pip install diffusers accelerate transformers")


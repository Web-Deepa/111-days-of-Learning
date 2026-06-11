#day53 :Data Augumentation
#Image augumenation
import albumentations as a
import nlpaug.augmenter.word as naw
import numpy as np
import cv2
import torch
from torchvision.transforms import v2

#1.Define pipeline
transform=a.compose([
    a.RandomCrop(width=256,height=256,p=1.0),
    a.HorizontalFlip(p=0.5),
    a.RandomBrightnessContrast(p=0.2),
    a.Rotate(limit=40,p=0.5)
])

#2.Load Image data using opencv
img=cv2.imread("C:\Users\cer\Desktop\WebLab-Deepa\wel.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BRGB2R)

#3.Apply augumentation
augumented=transform(img=img)
augumented_img=augumented["image"]
print(  "Augumented image:",augumented_img)


#Text Augumentation
# 1. Initialize 
text = "It is 53 days of learning."
aug = naw.SynonymAug(semantics_by_name=True)

# 2. Augment the text
augmented_text = aug.augment(text)
print(augmented_text) 

#Tabular Augumentation
# 1. Create a tabular data
X_train = np.random.rand(100, 5)
y_train = np.random.randint(0, 2, size=(100,))

# 2. Generate random Gaussian noise
noise = np.random.normal(0, 0.05, X_train.shape)

# 3. Combine original data and noisy data 
X_augmented = np.concatenate([X_train, X_train + noise], axis=0)
y_augmented = np.concatenate([y_train, y_train], axis=0)

#Using Pytorch
#1. Define pipeline 
transform_pipeline = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandAugment(num_ops=2, magnitude=9),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])


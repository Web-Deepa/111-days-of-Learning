#Day61:Image Augmentations
import numpy as np
import imgaug.augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img

#1.Dummy batch of images
images=np.zeros((4,120,120,3),dtype=np.uint8)

#2.Define augmentation sequence
seq=iaa.Sequential([
    iaa.Crop(px=(1,12),keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0,2.0))
])

#3.Apply augmentations
images_aug=seq(images=images)

# Using keras
#1.Data generation
datagen=ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5,1.6)
)
#2.Load and prepare Image
img=load_img('sari.jpeg')
X=img_to_array(img)
X=X.reshape((1,)+X.shape)

#3.Generate augmented images
for i,batch in enumerate(datagen.flow(X,batch_size=1,
                                      save_to_dir='preview',
                                      save_prefix='aug',
                                      save_format='jpeg')):
    if i>=5:
        break

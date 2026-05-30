# Convolutional Neural Network (CNN)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical

# 1. CNN on MNIST
print("CNN - MNIST Digits--")
# load data
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
# normalize
x_tr = x_tr.astype('float32') / 255.0
x_te = x_te.astype('float32') / 255.0
x_tr = x_tr[..., np.newaxis] # add channel dim
x_te = x_te[..., np.newaxis]

# one-hot encode labels
y_tr_oh = to_categorical(y_tr, 10)
y_te_oh = to_categorical(y_te, 10)
print(f"Train shape:{x_tr.shape}")
print(f"Test shape:{x_te.shape}")

# Build CNN
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    # conv block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # conv block 2 - FIXED: capitalized Conv2D
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # flatten and classify - FIXED: capitalized Flatten
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# FIXED: corrected categorical_crossentropy spelling
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()
history = cnn.fit(x_tr, y_tr_oh, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
loss, acc = cnn.evaluate(x_te, y_te_oh, verbose=0)
print(f"Test Accuracy:{acc:.3f}")
print(f"Test Loss:{loss:.3f}")

# 2. Visualize
print("Visualize Predictions---")
y_pred = np.argmax(cnn.predict(x_te[:16], verbose=0), axis=1)

plt.figure(figsize=(10, 4))
for i in range(16):
    plt.subplot(2, 8, i+1)
    plt.imshow(x_te[i].squeeze(), cmap='gray')
    color = 'green' if y_pred[i] == y_te[i] else 'red'
    plt.title(f"P:{y_pred[i]}", color=color, fontsize=10)
    plt.axis('off')
plt.suptitle("CNN Predictions (green=correct, red=wrong)")

plt.savefig("cnn.png")
plt.show()

# 3. Training History Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("history.png")
plt.show()

# 4. Visualize CNN filters detect
print("Visualize Convulation filters--")
# block1 filters
filters = cnn.layers[0].get_weights()[0]
print(f"Filter shape: {filters.shape}  (3x3, 1 channel, 32 filters)")

plt.figure(figsize=(10, 3))
for i in range(16):
    plt.subplot(2, 8, i+1)
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.suptitle("First 16 Conv Filters (3x3)")
plt.tight_layout()
plt.savefig("block1_filter.png")
plt.show()


# 5. CNN on CIFAR-10 (color images)
print("CNN — CIFAR-10 (color images)----")
(x_tr2, y_tr2), (x_te2, y_te2) = cifar10.load_data()
x_tr2 = x_tr2.astype('float32') / 255.0
x_te2 = x_te2.astype('float32') / 255.0
y_tr2_oh2 = to_categorical(y_tr2, 10)
y_te2_oh2 = to_categorical(y_te2, 10)

class_names = ['business', 'project', 'engineer', 'flower', 'plane',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Train shape: {x_tr2.shape}  ← color images (32x32x3)")

cnn2 = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

cnn2.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
history2 = cnn2.fit(
    x_tr2, y_tr2_oh2,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)
loss2, acc2 = cnn2.evaluate(x_te2, y_te2_oh2, verbose=0)
print(f"CIFAR-10 Test Accuracy : {acc2:.3f}")

# SECTION 6 — Compare: Dense NN vs CNN on MNIST
print(" Dense NN vs CNN — MNIST---")
# Simple Dense NN/no convulation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_flat_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_flat_test  = x_test.reshape(-1, 784).astype('float32')  / 255.0
y_train_oh   = to_categorical(y_train, 10)
y_test_oh    = to_categorical(y_test,  10)

dense_nn = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,  activation='relu'),
    layers.Dense(10,  activation='softmax')
])
dense_nn.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
dense_nn.fit(x_flat_train, y_train_oh,
             epochs=5, batch_size=64,
             validation_split=0.1, verbose=0)

_, acc_dense = dense_nn.evaluate(x_flat_test, y_test_oh, verbose=0)
print(f"{'Model':<20} {'Test Accuracy':>14}")
print("-" * 36)
print(f"{'Dense NN':<20} {acc_dense:>14.4f}")
print(f"{'CNN':<20} {acc:>14.4f}")
print(f"\n CNN wins by: +{acc - acc_dense:.4f}")

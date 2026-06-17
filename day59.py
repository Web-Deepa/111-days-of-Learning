
#  Day 59 — Custom Training Loops with tf.GradientTape
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1.Data
data = load_breast_cancer()
x, y = data.data, data.target.astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train).astype(np.float32)
x_test  = sc.transform(x_test).astype(np.float32)

#2.Pipelining
BATCH = 32
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
           .shuffle(500).batch(BATCH)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH)

#3.Model build
model = keras.Sequential([
    layers.Input(shape=(30,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1,  activation='sigmoid')
])

optimizer  = keras.optimizers.Adam(learning_rate=0.001)
loss_fn    = keras.losses.BinaryCrossentropy()

train_acc  = keras.metrics.BinaryAccuracy()
val_acc    = keras.metrics.BinaryAccuracy()

#4.Custom training
@tf.function   
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training=True)
        loss   = loss_fn(y_batch, y_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y_batch, y_pred)
    return loss

@tf.function
def test_step(x_batch, y_batch):
    y_pred = model(x_batch, training=False)
    val_acc.update_state(y_batch, y_pred)

#5.Custom training loop
print("Custom Training Loop:")
EPOCHS      = 30
train_losses = []
val_accs     = []

for epoch in range(EPOCHS):
    train_acc.reset_state()
    val_acc.reset_state()
    epoch_loss = []

    # Training
    for x_batch, y_batch in train_ds:
        loss = train_step(x_batch, y_batch)
        epoch_loss.append(float(loss))

    # Validation
    for x_batch, y_batch in test_ds:
        test_step(x_batch, y_batch)

    avg_loss = np.mean(epoch_loss)
    train_losses.append(avg_loss)
    val_accs.append(float(val_acc.result()))

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
              f"Train acc: {float(train_acc.result()):.3f} | "
              f"Val acc: {float(val_acc.result()):.3f}")

print(f"\nFinal val accuracy: {val_accs[-1]:.3f}")

#6.Loss funtion
print("Custom Loss Function:")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred   = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    bce      = -y_true*tf.math.log(y_pred) - (1-y_true)*tf.math.log(1-y_pred)
    weight   = alpha * y_true * tf.pow(1-y_pred, gamma) + \
               (1-alpha) * (1-y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * bce)

model2 = keras.Sequential([
    layers.Input(shape=(30,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,  activation='sigmoid')
])
model2.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])
h2 = model2.fit(x_train, y_train, epochs=30, batch_size=32,
                 validation_split=0.2, verbose=0)
print(f"Focal loss model accuracy: {model2.evaluate(x_test, y_test, verbose=0)[1]:.3f}")



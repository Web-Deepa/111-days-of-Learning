
#  Day 58 — Regularization (L1, L2, Dropout, Early Stopping)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1. Data
data = load_breast_cancer()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test)


def build_model(reg_type=None, dropout=0.0):#model building
    if   reg_type == 'l1': reg = regularizers.l1(0.01)
    elif reg_type == 'l2': reg = regularizers.l2(0.01)
    elif reg_type == 'l1l2': reg = regularizers.l1_l2(l1=0.01, l2=0.01)
    else: reg = None

    return keras.Sequential([
        layers.Input(shape=(30,)),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(64,  activation='relu', kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(1,   activation='sigmoid')
    ])

# 2.Compare regularization techniques
print("Regularization Comparison:")
configs = {
    'No regularization' : build_model(),
    'L2 regularization' : build_model(reg_type='l2'),
    'Dropout(0.3)'      : build_model(dropout=0.3),
    'L2 + Dropout'      : build_model(reg_type='l2', dropout=0.3),
}

histories = {}
print(f"{'Model':<25} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10}")
print("-" * 57)

for name, model in configs.items():
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x_train, y_train, epochs=100, batch_size=32,
                  validation_split=0.2, verbose=0)
    histories[name] = h
    tr_acc  = h.history['accuracy'][-1]
    val_acc = h.history['val_accuracy'][-1]
    te_acc  = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"{name:<25} {tr_acc:>10.3f} {val_acc:>10.3f} {te_acc:>10.3f}")

# 3.—Early Stopping
print("Early Stopping:")
model_es = build_model(reg_type='l2', dropout=0.3)
model_es.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,          
    restore_best_weights=True,
    verbose=1
)
h_es = model_es.fit(
    x_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)
print(f"Stopped at epoch : {len(h_es.history['loss'])}")
print(f"Test accuracy    : {model_es.evaluate(x_test, y_test, verbose=0)[1]:.3f}")


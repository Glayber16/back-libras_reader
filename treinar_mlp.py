import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

print("Carregando dataset")
try:
    df = pd.read_csv('dataset_libras.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'dataset_libras.csv' nao encontrado")
    exit()

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print(f"Formato dos dados para MLP: {X.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_
np.save('classes.npy', classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Iniciando treinamento da MLP")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save('modelo_mlp.keras')
print("Modelo salvo como 'modelo_mlp.keras'")

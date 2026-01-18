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

num_amostras = X.shape[0]
if num_amostras == 0:
    print("Erro: Dataset vazio")
    exit()

X_reshaped = X.reshape(num_amostras, 21, 3)
print(f"Formato dos dados para CNN: {X_reshaped.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_
np.save('classes.npy', classes)
print(f"Classes aprendidas: {classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_encoded, test_size=0.2, random_state=42
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(21, 3)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Iniciando treinamento da CNN 1D")
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save('modelo_cnn1d.keras')
print("Modelo salvo como 'modelo_cnn1d.keras'")

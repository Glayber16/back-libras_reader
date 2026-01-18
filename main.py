import cv2
import base64
import numpy as np
import mediapipe as mp
import json
import os
from fastapi import FastAPI, WebSocket
import uvicorn

import keras
from keras.models import load_model
print(f"Biblioteca Keras carregada (Versão: {keras.__version__})")

app = FastAPI()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

model_mlp = None
model_cnn = None
classes = ["A", "B", "C"]

print("Carregando modelos de IA...")

if os.path.exists("classes.npy"):
    try:
        classes = np.load("classes.npy", allow_pickle=True)
        print(f"Classes carregadas: {classes}")
    except Exception as e:
        print(f"Erro ao abrir classes.npy: {e}")

try:
    model_mlp = load_model("modelo_mlp.keras")
    print("Rede MLP carregada")
except Exception as e:
    print(f"Erro ao carregar modelo MLP: {e}")

try:
    model_cnn = load_model("modelo_cnn1d.keras")
    print("Rede CNN carregada")
except Exception as e:
    print(f"Erro ao carregar modelo CNN: {e}")


@app.websocket("/ws")
async def libras_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image")
            
            if image_b64:
                print(".", end="", flush=True)
            else:
                print("JSON recebido sem imagem")
            
            if image_b64:
                if ',' in image_b64:
                    _, imgstr = image_b64.split(',')
                else:
                    imgstr = image_b64
                
                nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Erro ao decodificar imagem")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                resposta_final = {
                    "mlp": {"letra": "-", "confianca": 0.0},
                    "cnn": {"letra": "-", "confianca": 0.0}
                }

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        pontos = []
                        for lm in hand_landmarks.landmark:
                            pontos.extend([lm.x, lm.y, lm.z])
                        
                        dados_brutos = np.array(pontos)

                        if model_mlp:
                            try:
                                input_mlp = dados_brutos.reshape(1, 63)
                                pred_mlp = model_mlp.predict(input_mlp, verbose=0)
                                idx_mlp = np.argmax(pred_mlp)
                                conf_mlp = float(np.max(pred_mlp))
                                resposta_final["mlp"] = {
                                    "letra": str(classes[idx_mlp]),
                                    "confianca": round(conf_mlp * 100, 1)
                                }
                            except Exception as e:
                                print(f"Erro MLP: {e}")

                        if model_cnn:
                            try:
                                input_cnn = dados_brutos.reshape(1, 21, 3)
                                pred_cnn = model_cnn.predict(input_cnn, verbose=0)
                                idx_cnn = np.argmax(pred_cnn)
                                conf_cnn = float(np.max(pred_cnn))
                                resposta_final["cnn"] = {
                                    "letra": str(classes[idx_cnn]),
                                    "confianca": round(conf_cnn * 100, 1)
                                }
                            except Exception as e:
                                print(f"Erro CNN: {e}")

                await websocket.send_json(resposta_final)

    except Exception as e:
        print(f"Conexão encerrada: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

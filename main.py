import cv2
import base64
import numpy as np
import mediapipe as mp
import os
from fastapi import FastAPI, WebSocket
import uvicorn
from keras.models import load_model

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

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

if os.path.exists("classes.npy"):
    classes = np.load("classes.npy", allow_pickle=True)

try:
    model_mlp = load_model("modelo_mlp.keras")
    print("MLP carregada")
except:
    print("Erro ao carregar MLP")

try:
    model_cnn = load_model("modelo_cnn1d.keras")
    print("CNN carregada")
except:
    print("Erro ao carregar CNN")

@app.websocket("/ws")
async def libras_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image")

            if not image_b64:
                continue

            if ',' in image_b64:
                _, imgstr = image_b64.split(',')
            else:
                imgstr = image_b64

            nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            resposta = {
                "mlp": {"letra": "-", "confianca": 0},
                "cnn": {"letra": "-", "confianca": 0},
                "roi": None
            }

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]

                    x_min = min(x_vals) * 100
                    y_min = min(y_vals) * 100
                    box_w = (max(x_vals) - min(x_vals)) * 100
                    box_h = (max(y_vals) - min(y_vals)) * 100

                    resposta["roi"] = {
                        "x": max(0, x_min - 5),
                        "y": max(0, y_min - 5),
                        "w": box_w + 10,
                        "h": box_h + 10
                    }

                    pontos = []
                    for lm in hand_landmarks.landmark:
                        pontos.extend([lm.x, lm.y, lm.z])
                    dados = np.array(pontos)

                    if model_mlp:
                        p = model_mlp.predict(dados.reshape(1, 63), verbose=0)
                        resposta["mlp"] = {
                            "letra": str(classes[np.argmax(p)]),
                            "confianca": float(np.max(p)) * 100
                        }

                    if model_cnn:
                        p = model_cnn.predict(dados.reshape(1, 21, 3), verbose=0)
                        resposta["cnn"] = {
                            "letra": str(classes[np.argmax(p)]),
                            "confianca": float(np.max(p)) * 100
                        }

            await websocket.send_json(resposta)

    except Exception as e:
        print("Desconectado", e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

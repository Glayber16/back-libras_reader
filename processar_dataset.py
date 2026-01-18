import cv2
import mediapipe as mp
import csv
import os
import numpy as np

CAMINHO_DATASET = r'dataset\train' 
ARQUIVO_SAIDA = 'dataset_libras.csv'


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

if not os.path.exists(ARQUIVO_SAIDA):
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    
    with open(ARQUIVO_SAIDA, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

print(f"🔄 Lendo imagens de: {CAMINHO_DATASET}")
contador = 0

for nome_pasta in sorted(os.listdir(CAMINHO_DATASET)):
    caminho_letra = os.path.join(CAMINHO_DATASET, nome_pasta)
    
    if os.path.isdir(caminho_letra):
        print(f"📂 Processando letra {nome_pasta}...")
        
        for nome_foto in os.listdir(caminho_letra):
            caminho_foto = os.path.join(caminho_letra, nome_foto)
            
            try:
                img = cv2.imread(caminho_foto)
                if img is None: continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        row = [nome_pasta] 
                        for lm in hand_landmarks.landmark:
                            row.extend([lm.x, lm.y, lm.z])
                        
                        with open(ARQUIVO_SAIDA, mode='a', newline='') as f:
                            csv.writer(f).writerow(row)
                        
                        contador += 1
                        if contador % 100 == 0:
                            print(f"   -> {contador} imagens processadas...")

            except Exception as e:
                print(f"Erro na imagem {nome_foto}: {e}")

print(f"✅ Concluído! Total de {contador} amostras salvas em '{ARQUIVO_SAIDA}'")
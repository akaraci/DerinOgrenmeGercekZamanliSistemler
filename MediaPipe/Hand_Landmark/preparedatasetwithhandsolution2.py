# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:46:50 2024

@author: akara
"""

import cv2
import mediapipe as mp
import csv

# MediaPipe Hands modülünü başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Kamera açma
cap = cv2.VideoCapture(0)

framecount=0
# CSV dosyasını başlat
with open('hand_landmarks_data.csv', 'w', newline='') as csvfile:
    # Başlıkları oluştur
    fieldnames = []
    for hand in ['Left', 'Right']:
        for i in range(21):  # 21 landmark
            fieldnames.extend([f'{hand}_landmark_{i}_x', f'{hand}_landmark_{i}_y', f'{hand}_landmark_{i}_z'])
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü RGB'ye çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # El tespiti
    results = hands.process(frame_rgb)

    # Veri kaydetmek için satır oluştur. Dictionary key fieldnames, value none
    row_data = {field: None for field in fieldnames}

    if results.multi_hand_landmarks and results.multi_handedness:
        framecount+=1
        #veri almaya başlamadan önce 50 frame bekle
        if (framecount>=50): 
            if (framecount==50):print("Başladı...")
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Elin sağ mı sol mu olduğunu tespit et
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                hand_label = 'Right' if hand_label == 'Left' else 'Left'
    
                # Landmark bilgilerini ekle
                for i, landmark in enumerate(hand_landmarks.landmark):
                    row_data[f'{hand_label}_landmark_{i}_x'] = landmark.x
                    row_data[f'{hand_label}_landmark_{i}_y'] = landmark.y
                    row_data[f'{hand_label}_landmark_{i}_z'] = landmark.z
    
            # CSV dosyasına yaz
            with open('hand_landmarks_data.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)
            if (framecount>=79):
                print("bitti")
                break
    # Görüntüde göster (isteğe bağlı)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

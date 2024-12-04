# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:00:29 2024

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

# Veri seti için CSV dosyasının başlıklarını yazalım
with open('hand_landmarks_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['hand', 'landmark_id', 'xl', 'yl', 'zl','hand', 'xr', 'yr', 'zr']
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
    
    if results.multi_hand_landmarks:
        #landmark nesne dizisi sırayla hand_landmarks'a ve indeksi de hand_idx'e aktarılıyor
        xl,yl,zl,xr,yr,zr=0,0,0,0,0,0
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            #ilgili verinin hangi ele ait olduğu belirleniyor.
            if results.multi_handedness[hand_idx].classification[0].label == 'Right': 
                hand_label = 'Left'  
            else: hand_label = 'Right'
            
            # hand_landmarks.landmark nesne dizisindeki nesneleri sırayla landmark değişkenne aktar ve koordinatları bu nesne üzerinden elde et
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                if (hand_label == 'Left'):
                    xl = landmark.x
                    yl = landmark.y
                    zl = landmark.z
                else:
                    xr = landmark.x
                    yr = landmark.y
                    zr = landmark.z 
                               
                # Görüntüye çizme (isteğe bağlı)
                height, width, _ = frame.shape
                x_pos, y_pos = int(x * width), int(y * height)
                cv2.circle(frame, (x_pos, y_pos), 5, (0, 255, 0), -1)
                
                # Landmark id yazma (isteğe bağlı)
                cv2.putText(frame, str(i), (x_pos + 5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # CSV dosyasına yazma
            with open('hand_landmarks_data.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'hand': hand_label, 'landmark_id': i, 'xl': xl, 'yl': yl, 'zl': zl})
    
    # Sonuçları göster
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:18:04 2024

@author: akara
"""

import cv2
import mediapipe as mp

# MediaPipe Hands modülünü başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Kamera açma
cap = cv2.VideoCapture(0)
framecount=0
left,right=[],[]
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:break
    # Görüntüyü BGR'den RGB'ye dönüştür
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # El ve landmark tespiti
    results = hands.process(frame_rgb)
    
    # Eğer eller tespit edildiyse
    if results.multi_hand_landmarks:
        print("results.multi_hand_landmarks=",results.multi_hand_landmarks)
        print("results.multi_handedness=",results.multi_handedness)
        print("Landmark sayısı=",len(results.multi_hand_landmarks))
        for hand_landmarks,handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            print(type(hand_landmarks))
            sample=[]
            framecount+=1
            if (framecount>=50): # frame oluncaya kadar bekle. İlgili işareti kullanıcının ayarlaması için zaman verir.
                #50 frame'den sonra veri setini oluşturmaya başla.
                for i in range(21):#21 noktayı sırayla aktar. her bir nokta bir landmark nesnesidir
                    lm=hand_landmarks.landmark[i] #i.nesneyi al
                    print("ladmark-",i,"=",lm)
                    sample.append((lm.x,lm.y,lm.z))
                print("Sample=",sample)
                
                #verinin hangi ele ait olduğu belirleniyor
                #selfie olduğu için index'leri ters değerlendiriyoruz.
                #handedness bir tane classification nesnesi içerir.
                #bu nedenle handedness.classification[0] ilk nesneyi ifade eder
                if(handedness.classification[0].index==1): 
                    left.append(sample)
                if(handedness.classification[0].index==0):
                    right.append(sample)
            
                
                print("Örnekteki veri sayısı=",len(sample))
                print("Sol ele ait toplam örnek=",len(left))
                print("Sağ ele ait toplam örnek=",len(right))
            
            # Her bir elin işaret noktalarını çiz
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Sonuçları görüntüle
    cv2.imshow('Hand Tracking', frame)
    
    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q') or len(left)==30 or len(right)==30:
        break

cap.release()
cv2.destroyAllWindows()
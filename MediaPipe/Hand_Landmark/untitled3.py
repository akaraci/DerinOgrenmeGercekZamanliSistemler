# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:05:27 2023

@author: akara
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

global results
cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    #print(results.__doc__)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        #el bilgisi ve ilgili ele ait landmark bilgileri sırayla aktarılıyor
        for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
        
            print("hand_landmarks:",hand_landmarks) #eldeki 21 noktayı liste olarak verir
            print("landmark[0].x:",hand_landmarks.landmark[0].x) #eldeki 21 noktayı liste olarak verir
            
            print("handedness:",handedness)  #tespit edilen eli verir
            
            #selfieden dolayı sağ ve sol el yer değiştirdiği için el bilgisi ayarlanıyor
            if (handedness.classification[0].label=="Left"):print("Sağ el") 
            else: print("Sol el")
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
#quit()
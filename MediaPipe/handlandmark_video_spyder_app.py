# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:42:15 2023

@author: akara
"""

import cv2
#from google.colab.patches import cv2_imshow

#%%-----------Visualization utilities
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

#%%
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

base_options = python.BaseOptions(model_asset_path='c:/hand_landmarker.task')
x,y,z=[],[],[]


hand_landmarks_left=[]
hand_landmarks_right=[]
adet=0

def print_result(result, output_image: mp.Image, timestamp_ms: int):      
    if result.hand_landmarks:
        hand_landmarks_list = result.hand_landmarks
        handedness_list = result.handedness #el bilgileri
        # print("el sayısı=",len(handedness_list))
        # print("eller=",handedness_list)
        # print("Hand landmark list:",hand_landmarks_list)
        #el ve ladmark için veriler sırayla aktarılıyor
        for handlandmark,hand in zip(hand_landmarks_list,handedness_list): #her bir el için landmark bilgilerini sırayla aktarır            
            #print(handlandmark[0].x)
            #print(handlandmark.NormalizedLandmark[0].x)
            sample=[]
            for i in range(21): #i her bir parmaklardaki landmark göstergesidir. listenin 21 elemanı var. Her biri nesne
                sample.append((handlandmark[i].x,handlandmark[i].y,handlandmark[i].z))
                print(handlandmark[i].x)
            if (hand[0].index==1):#listenin 1 elemanı var o da bir nesne bu nedenle 0. elemana bakılıyor
                hand_landmarks_left.append(sample) #eldeki 0 indexli noktanın x,y ve z koordinatı
            if (hand[0].index==0):
                hand_landmarks_right.append(sample)                         
                        
            print("left=",hand_landmarks_left) #verileri görmek isterseniz açın, istemezseniz kapatın.
            print ("Sol el için Alınan Örnek Sayısı=",len(hand_landmarks_left)) #örnek sayısı 30 olduğunda aşağıda kontrol edilip çıkılıyor
            print("Sağ El için alınan örnek sayısı=",len(hand_landmarks_right))
        
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow("Hand Landmark",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    #x'e basılınca ya da 30 örnek alınınca çık
    if cv2.waitKey(50) & 0xFF == ord('x') or len(hand_landmarks_left)==30 or len(hand_landmarks_right)==30 : 
        print("Sol el için ilk örnek veri:",hand_landmarks_left[0])
        print("Sağ el için ilk örnek veri:",hand_landmarks_right[0])
        cap.release()
        cv2.destroyAllWindows()
        quit(0)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='c:\hand_landmarker.task'),num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,30)

frame_count=0
while (True):
    
    _,image = cap.read() 

    if image is not None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        detection_result = detector.detect_async(mp_image, timestamp_ms)

    if cv2.waitKey(50) & 0xFF == ord('x'): 
        cap.release()
        cv2.destroyAllWindows()      
        break
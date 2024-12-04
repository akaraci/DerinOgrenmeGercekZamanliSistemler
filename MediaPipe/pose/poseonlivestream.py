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


#%% bulunan landmark'ları resim üzerine yerleştirir

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

#%%
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import csv


adet=0

def print_result(result, output_image: mp.Image, timestamp_ms: int):
    global adet
    if result.pose_landmarks:
        normalizelandmarkList = result.pose_landmarks
        print("Vücut sayısı=",len(normalizelandmarkList))
        #birden fazla kişi varsa her kişiye ait landmarkleri sırayla aktar
        for normalizelanmark in normalizelandmarkList: 
            print(len(normalizelanmark)) #birinci kişinin landmark bilgileri (33 adet)
            row={column:None for column in columns}
            #33 adet landmark'a tek, tek eriş. her eleman NormalizedLandmark nesnesidir
            for i,landmark in enumerate(normalizelanmark): 
                #print(landmark)
                i*=3 #i=0 için->0, i=1 için 3'den başlayacak 
                row[columns[i]]=landmark.x
                row[columns[i+1]]=landmark.y
                row[columns[i+2]]=landmark.z

            adet=adet+1
            print("alınan veri:",adet)
            #data'lar csv dosaysına yazdırılıyor
            with open('pose_landmarks_data.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writerow(row)

    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow("Hand Landmark",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    #x'e basılınca ya da 30 örnek alınınca çık
    if cv2.waitKey(50) & 0xFF == ord('x') or adet==30 : 
        cap.release()
        cv2.destroyAllWindows()
        quit(0)


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path="c:\\pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,30)

labels=["nose","left eye (inner)","left eye", "left eye (outer)","right eye (inner)","right eye","right eye (outer)",
        "left ear","right ear","mouth (left)","mouth (right)","left shoulder","right shoulder","left elbow",
        "right elbow","left wrist","right wrist","left pinky","right pinky","left index","right index","left thumb",
        "right thumb","left hip","right hip","left knee","right knee","left ankle","right ankle","left heel","right heel",
        "left foot index","right foot index"]

columns=[]
for label in labels:
    columns.extend([f"{label}_x",f"{label}_y",f"{label}_z"])

with open('pose_landmarks_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()
    
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
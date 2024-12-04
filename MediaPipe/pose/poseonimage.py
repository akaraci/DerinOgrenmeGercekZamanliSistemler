# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:34:27 2024

@author: akara
"""
#%%
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

img = cv2.imread("imagepose.jpg")
cv2.imshow("Orginal Image",img)
cv2.waitKey(0)
#%%
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import csv

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
    
#Diğer modelleri indirebilirsiniz
#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
base_options = python.BaseOptions(model_asset_path='c:\\pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True) #output_segmentation_masks=True->kişiyi segmente etmek için kullanılır

detector = vision.PoseLandmarker.create_from_options(options)

image = mp.Image.create_from_file("imagepose.jpg")

detection_result = detector.detect(image)

worldlandmarkList=detection_result.pose_world_landmarks
normalizelandmarkList=detection_result.pose_landmarks
#0.vücudun, 0.noktasının(nose) x değeri
print(normalizelandmarkList[0][0].x)

#birden fazla kişi varsa her kişiye ait landmarkleri sırayla aktar
for normalizelanmark in normalizelandmarkList: 
    print(len(normalizelanmark)) #birinci kişinin landmark bilgileri (33 adet)
    data={column:None for column in columns}
    #33 adet landmark'a tek, tek eriş. her eleman NormalizedLandmark nesnesidir
    for i,landmark in enumerate(normalizelanmark): 
        #print(landmark)
        i*=3 #i=0 için->0, i=1 için 3'den başlayacak 
        data[columns[i]]=landmark.x
        data[columns[i+1]]=landmark.y
        data[columns[i+2]]=landmark.z      
        #☺row.add(data,axis=1)

print(data)
#data'lar csv dosaysına yazdırılıyor
with open('pose_landmarks_data.csv', 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writerow(data)

annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Pose Image",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

#%%
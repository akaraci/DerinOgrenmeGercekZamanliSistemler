# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 08:37:15 2024

@author: akara
"""

#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  #print(face_landmarks_list)
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()
  
#%%
import cv2
img = cv2.imread("image.png")

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VisionRunningMode = mp.tasks.vision.RunningMode
# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='c:\\face_landmarker.task') #'face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1,
                                       running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while (True):
    _,image = cap.read() 
    annotated_image_manual=image #üzerine noktalar manual eklenecek
    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect_for_video(image,int(cap.get(cv2.CAP_PROP_POS_MSEC))) #Tüm tespit sonuçarı
    
   #------manual görselleştirme 
    face_landmarks_list = detection_result.face_landmarks #Tüm yüzlere ait yüz yer işaretlerini al
    firstface_landmark_list=face_landmarks_list[0]  #tespit edilen ilk yüze ait 478 yer işareti x,y,z koordinatı
    height, width, _ = annotated_image_manual.shape #yüz yer işaretlerini resim üzerine yerleştirirken noktaları image'a uyarlamak için kullanılacak.
    x_coordinates = [landmark.x for landmark in firstface_landmark_list] #tüm x'leri listeye aktar
    y_coordinates = [landmark.y for landmark in firstface_landmark_list] #tüm y'leri listeye aktar.
    for x,y in zip(x_coordinates,y_coordinates): #yüzün üzerine noktaları yerleştir.
        print(x*width)
        cv2.circle(img=annotated_image_manual, center=(int(x*width),int(y*height)), radius=2, color=(0,255,0),thickness=-1)
    #-----------------------
    # STEP 5: Görselleştirme media pipe ile yapılıyor.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    cv2.imshow("media pipe image",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("manual image",cv2.cvtColor(annotated_image_manual, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(50) & 0xFF == ord('x'): #x çıkış
        cap.release()
        cv2.destroyAllWindows()
        break

#%%
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
    #print('hand landmarker result: {}'.format(result)) #elde edilen yer işaretlerini yazdır
    #print(result.hand_landmarks)
    #---------Görüntü üzerinde yer işaretlerini göster
    # x.append(result.hand_landmarks.landmark.x)
        
    if result.hand_landmarks:
        #print(result.hand_landmarks[0].NormalizedLandmark.x)
        hand_landmarks_list = result.hand_landmarks
        handedness_list = result.handedness #el bilgileri
        print("el sayısı=",len(handedness_list))
        #print(handedness_list)
        print("eller=",handedness_list)
        #print("değer=",len(hand_landmarks_list))
        # Loop through the detected hands to visualize.
        #print(adet)
        for hand in handedness_list: #her bir el için landmark'ler alınacak
            for idx in range(len(hand_landmarks_list)): #her bir el için landmark bilgilerini sırayla aktarır      
                #print("index=",hand[0].index)            
                sample=[]
                for i in range(21): #i her bir parmaklardaki landmark göstergesidir
                    sample.append((hand_landmarks_list[idx][i].x,hand_landmarks_list[idx][i].y,hand_landmarks_list[idx][i].z))
                if (hand[0].index==1): #sol else
                    hand_landmarks_left.append(sample) #eldeki 0 indexli noktanın x,y ve z koordinatı
                if (hand[0].index==0): #ilgili el bilgisindeki ilk elemanı(Category tipinde) elde ediyor. 
                    hand_landmarks_right.append(sample)                         
                            
                print("left=",hand_landmarks_left) #verileri görmek isterseniz açın, istemezseniz kapatın.
                #print("rigth=",hand_landmarks_right)
                print ("Sol el için Alınan Örnek Sayısı=",len(hand_landmarks_left)) #örnek sayısı 30 olduğunda aşağıda kontrol edilip çıkılıyor
                print("Sağ El için alınan örnek sayısı=",len(hand_landmarks_right))
                
            #for hand_landmarks in result.hand_landmarks:
                #print([landmark for landmark in hand_landmarks])
                # print("değer=",hand_landmarks[0])
                # print(type(hand_landmarks[0]))
    else:
        print("El işareti algılanamadı.")
    # if (count>=1000000):quit()
    # print([landmark.x for landmark in result.hand_landmarks])
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow("Hand Landmark",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    #x'e basılınca ya da 30 örnek alınınca çık
    if cv2.waitKey(50) & 0xFF == ord('x') or len(hand_landmarks_left)==30 or len(hand_landmarks_right)==30 : 
        cap.release()
        cv2.destroyAllWindows()
        quit()


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='c:\hand_landmarker.task'),num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)


IMAGE_FILE = 'savedframe.jpg'
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,30)

while (True):
    
    _,image = cap.read() 
    #print(cv2.CAP_PROP_FPS)
    #cv2.imwrite("savedframe.jpg",frame) #her frame'i kaydet
    #img=cv2.imread("savedframe.jpg")
    #image = mp.Image.Create(image)
    #image.flags.writeable = False
    #image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    #image.flags.writeable = False
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    #image.flags.writeable = False
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #print("Time",cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = detector.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    #print("adet=",adet)
    #print(detection_result)
    #image.flags.writeable = True
    #print(detection_result)
    #annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    #cv2.imshow("Hand Landmark",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    #cv2.imshow(winname="Face",mat=img)
    if cv2.waitKey(50) & 0xFF == ord('x'): #x çıkış
        cap.release()
        cv2.destroyAllWindows()      
        break

# cap.release()
# cv2.destroyAllWindows()




# # STEP 2: Create an HandLandmarker object.



# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")

# # STEP 4: Detect hand landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the classification result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow("Hand Landmark",cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
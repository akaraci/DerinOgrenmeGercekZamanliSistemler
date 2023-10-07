# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:42:16 2023

@author: akara
"""
#Real-time Object Detecto with MediaPipe
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow


import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image


IMAGE_FILE = 'savedframe.jpg'
cap = cv2.VideoCapture(0)

while (True):
    
    _,frame = cap.read() 

    cv2.imwrite("savedframe.jpg",frame) #her frame'i kaydet
    #object detector ayarlarını yap 
    #efficientdet.tflite dosyasını c'ye kopayalayın. Masaüstünde olunca klasörden görmüyor
    #https://developers.google.com/mediapipe/solutions/vision/object_detector/index#models
    base_options = python.BaseOptions(model_asset_path='/ssd_mobilenet_v2.tflite') #efficientdet.tflite #efficientdet_lite2.tflite
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                            score_threshold=0.5)
    #detector nesnesini ilgili ayarlarla oluştur
    detector = vision.ObjectDetector.create_from_options(options)
    
    # kaydedilen her frame'i model formatına çevir
    image = mp.Image.create_from_file(IMAGE_FILE)
    
    # Detect objects in the input image.
    #boundingbox (x,y,width,height)
    detection_result = detector.detect(image)
    
    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    #bounding box ile resmi birleştir
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    #Birleştirilmiş sonuç resmini göster
    cv2.imshow(winname="Face",mat=annotated_image)
    if cv2.waitKey(50) & 0xFF == ord('x'): #x çıkış
        break
cap.release()
cv2.destroyAllWindows()


#%%
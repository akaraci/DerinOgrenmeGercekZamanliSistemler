# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:07:40 2023

@author: akara
"""

#dlib yüz bölgesinde farklı noktaları bize vermektedir ve aşağıdaki gibi bölgeler kodlanmıştır. 
#Her numaraya karşılık gelen x ve y koordinatları elde edilebilmektedir.
# Çene Noktaları: 0-16
# Sağ Kaş Noktaları: 17-21
# Sol Kaş Noktaları: 22-26
# Burun Noktaları: 27-35
# Sağ Göz Noktaları: 36-41
# Sol Göz Noktaları: 42-47
# Ağız Noktaları: 48-60
# Dudak Noktaları: 61-67
#%% Bu uygulama resimden yüz bölgesini tespit etmek için kullanılmaktadır
#%%
import cv2
import dlib
import numpy as np
img = cv2.imread("foto.jpg")
# Resimi grayscale dönüştürür.
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

height = img.shape[0]
width = img.shape[1]

detector = dlib.get_frontal_face_detector()
faces = detector(gray)
print(faces)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yk=[]
xk=[]
#resim boyutunda tüm değerleri sıfır olan dizi oluşturuluyor
mask = np.zeros((height, width), dtype=np.uint8) 

for face in faces:  
    landmarks = predictor(image=gray, box=face)
    for i in range(36,42):
        print(landmarks.part(i).x,end=",")
        xk.append([landmarks.part(i).x,landmarks.part(i).y]) 
    points = np.array([xk])
    #göz bölgesinin pointleri kullanılarak ilgili bölge 255 değerine set ediliyor.
    cv2.fillPoly(mask, points, (255))
#resim ve mask and işlemine tabi tutuluyor
res = cv2.bitwise_and(img,img,mask = mask)
rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped_image = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

from matplotlib import pyplot as plt
plt.imshow(img)
plt.show()
plt.imshow(cropped_image)
plt.show()










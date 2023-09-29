# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:05:37 2023

@author: akara
"""

import numpy as np
import cv2

img = cv2.imread("foto.jpg")
height = img.shape[0]
width = img.shape[1]

#Resim boyutunda değerleri sıfır olan array oluşturuluyor
mask = np.zeros((height, width), dtype=np.uint8)

#croplanacak Noktalar belirleniyor
points = np.array([[[100,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])

#göz bölgesinin pointleri kullanılarak ilgili bölge 255 değerine set ediliyor.
cv2.fillPoly(mask, points, (255))

#resim ve mask and işlemine tabi tutuluyor
res = cv2.bitwise_and(img,img,mask = mask)

rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

cv2.imshow("cropped" , cropped )
cv2.imshow("same size" , res),
cv2.waitKey(0)



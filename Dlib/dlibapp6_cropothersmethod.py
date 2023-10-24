# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:51:41 2023

@author: akara
"""

from imutils import face_utils
import cv2
import dlib
import numpy as np

img = cv2.imread("foto.jpg")
# Resimi grayscale dönüştürür.
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

#Yüz bölgesinin koordinatları tespit ediliyor
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

for face in faces:
    
    #yüzdeki bölgelerin(göz, dudak, ağız vb.) koordinatları elde ediliyor.
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    landmarks = predictor(image=gray, box=face)
    
    #sağ(42-47) ve sol göz(36-41) için noktaları tutan dizi indisleri alınıyor ()
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    #sağ ve sol gözün koorddinatları elde ediliyor 
    landmarks = face_utils.shape_to_np(landmarks)#koordinatlar numpy array'e dönüştürülüyor.
    leftEye = landmarks[lStart:lEnd] #sol göz koordinatları alınıyor
    rightEye = landmarks[rStart:rEnd] #sağ göz koordinatları alınıyor
    
    #gözün alt ve üst değerini alıp yüksekliiğini hesapla
    l_uppery = min(leftEye[1:3,1]) #Dlib indis-->37 ve 38
    l_lowy = max(leftEye[4:,1])    ##Dlib indis-->40 ve 41
    l_dify = abs(l_uppery - l_lowy)
    
    #gözün genişliğini hesapla
    lw = (leftEye[3][0] - leftEye[0][0]) #Dlib indis-> 39 ve 36
    
    #26x34 boyutlarında kırp
    # minxl = (leftEye[0][0] - ((34-lw)/2))
    # maxxl = (leftEye[3][0] + ((34-lw)/2)) 
    # minyl = (l_uppery - ((26-l_dify)/2))
    # maxyl = (l_lowy + ((26-l_dify)/2))
    
    #genişlik ve yüksekliğin yarısı kadar kırpılacak karesel bölgeyi genişlet
    minxl = (leftEye[0][0]-lw/2) #ssol bitiş noktası(x)
    maxxl = (leftEye[3][0]+lw/2) #Sağ bitiş noktası(x)
    minyl = (l_uppery - l_dify/2) #gözün üst kısmı en üst koordinat(y)
    maxyl = (l_lowy + l_dify/2)   #gözün alt kısmı en alt kooridnat (y)
    
    # minxl = leftEye[0][0]
    # maxxl = leftEye[3][0]
    # minyl = l_uppery
    # maxyl = l_lowy
    
    #kareden göz kısmını kırp
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    from matplotlib import pyplot as plt
    plt.imshow(left_eye_image)
    cv2.waitKey(delay=0) #bir tuşa basıncaya kadar bekle
    cv2.destroyAllWindows() #bütün pencereleri kapat
    
    
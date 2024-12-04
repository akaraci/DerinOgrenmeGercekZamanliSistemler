# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:46:53 2023

@author: akara
"""

#!/usr/bin/env python

import cv2
cap = cv2.VideoCapture(0)

# cap = cap.set(cv2.CAP_PROP_FPS, 30) 
# #ret,frame = cap.read()
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)

cap.set(3, 768)
cap.set(4, 1024)

while(cap.isOpened()):

    cap.set(cv2.CAP_PROP_FPS,2) 
    ret,frame = cap.read()
    print(cap.get(cv2.CAP_PROP_FPS))
    if not ret:
        break

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

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

#Yüz bölgesini tespit edebilmemiz için get_frontal_face_detector fonksiyonunu kullanıyoruz.
#shape_predictor() fonksiyonu yüzün belirli özelliklerini bize sağlar.
detector = dlib.get_frontal_face_detector()
faces = detector(gray)
print(faces)
#nternette bir sürü model bulunmaktadır."shape_predictor_68_face_landmarks.dat" modelini kullandık

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    
    #Buradaki predictor fonksiyonu, yüz şemasına uyan 68 noktanın tümünü içeren bir nesne 
    #döndürecektir. Dikkat ederseniz, 27 noktası tam olarak gözlerin arasına denk gelmektedir. 
    #Her şey doğru çalıştıysa, yüzdeki gözler arasında yeşil bir nokta görmelisiniz.
    landmarks = predictor(image=gray, box=face)
    x = landmarks.part(27).x
    y = landmarks.part(27).y
    cropped_image = img[y1:y2, x1:x2]
    cv2.imshow(winname="Face",mat=cropped_image)
    cv2.circle(img=img, center=(x,y), radius=5, color=(0,255,0),thickness=-1)
    #Bir dikdörtgen çiziyoruz
    cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=4)


#resimi göster
im1=cv2.imshow(winname="CropFace",mat=cropped_image)
cv2.imshow(winname="Face",mat=img)

cv2.waitKey(delay=0) #bir tuşa basıncaya kadar bekle

cv2.destroyAllWindows() #bütün pencereleri kapat








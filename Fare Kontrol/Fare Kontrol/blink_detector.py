import tensorflow as tf
import cv2
import dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils
import pyautogui


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')


# yüzü algıla
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("haar Cascade xml dosyası yüklenirken hata."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    

    # yüz bulunmazsa diziyi 0 değeriyle yolla
    if len(rects) == 0:
        return []


    rects[:, 2:] += rects[:, :2]

    return rects

def cropEyes(frame):
	 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#gri resimden yüzü algıla
	te = detect(gray, minimumFeatureSize=(80, 80))

	#eğer yüz algılanmazsa 0 değeri döndür
	#1den fazla yüz algılanırsa farklı bir değerle tekrar işlem yap
	if len(te) == 0:
		return None
	elif len(te) > 1:
		face = te[0]
	elif len(te) == 1:
		[face] = te

	#kareden sadece yüz ksımını tut
	face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))
	
	#yüz kısmı için işaretlerini belirleyin
	shape = predictor(gray, face_rect)
	shape = face_utils.shape_to_np(shape)

	#sağ ve sol göz'ün indexleriini al
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	#sağ ve sol gözün koorddinatlarını çıkar
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	#gözün alt ve üst değerini alıp yüksekliiğini hesapla
	l_uppery = min(leftEye[1:3,1])
	l_lowy = max(leftEye[4:,1])
	l_dify = abs(l_uppery - l_lowy)

	#gözün genişliğini hesapla
	lw = (leftEye[3][0] - leftEye[0][0])

	#cnn için görüntünün(26, 34) olmasını istiyoruz
	#bu yüzden x ve y noktalarındaki farkın yarısını topladık
	# genişlikten yükseklik sırasıyla left-right ekseni
	# ve up-down
	minxl = (leftEye[0][0] - ((34-lw)/2))
	maxxl = (leftEye[3][0] + ((34-lw)/2)) 
	minyl = (l_uppery - ((26-l_dify)/2))
	maxyl = (l_lowy + ((26-l_dify)/2))
	
	#kareden göz kısmını kırp
	left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
	left_eye_rect = left_eye_rect.astype(int)
	left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]
	
	#yukardaki işlemleri sağ göz için de yapıyoruz
	r_uppery = min(rightEye[1:3,1])
	r_lowy = max(rightEye[4:,1])
	r_dify = abs(r_uppery - r_lowy)
	rw = (rightEye[3][0] - rightEye[0][0])
	minxr = (rightEye[0][0]-((34-rw)/2))
	maxxr = (rightEye[3][0] + ((34-rw)/2))
	minyr = (r_uppery - ((26-r_dify)/2))
	maxyr = (r_lowy + ((26-r_dify)/2))
	right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
	right_eye_rect = right_eye_rect.astype(int)
	right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

	#sol veya sağ gözü algılamazsa boş değeri döndür
	if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
		return None
	# yeniden boyutlandırma
	left_eye_image = cv2.resize(left_eye_image, (34, 26))
	right_eye_image = cv2.resize(right_eye_image, (34, 26))
	right_eye_image = cv2.flip(right_eye_image, 1)
	#sol gözü ve gözü gönder
	return left_eye_image, right_eye_image 

#fotoğrafı eğitimdeki formata çevirme
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

def main():
	#kamerayı aç
	camera = cv2.VideoCapture(0)
	font_type = cv2.FONT_HERSHEY_COMPLEX
	kalibre = 0
	model = load_model('blinkModel.hdf5')

	farex = 1920 / 2
	farey = 1080 / 2

	#blinks,değerleri toplam kırpma sayısıdır
	#close_counter, ardışık yakın tahminler için sayaç
	# mem_counter önceki döngünün sayacı
	close_counter_r = blinks_r = mem_counter_r= 0
	close_counter_l = blinks_l = mem_counter_l = 0
	state_l = ''
	state_r = ''

	font_type = cv2.FONT_HERSHEY_COMPLEX

	while True:
		ret, frame = camera.read()
		frame = cv2.flip(frame, 1)
		#gözleri algıla
		eyes = cropEyes(frame)


		if eyes is None:
			continue
		else:
			left_eye,right_eye = eyes

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		#iki gözün tahminlerinin ortalaması
		prediction_r = (model.predict(cnnPreprocess(right_eye)))
		prediction_l = (model.predict(cnnPreprocess(left_eye)))

		#gözler açıksa kapalı göz için sayacı sıfırlayın
		if prediction_r > 0.5 :
			state_r = 'Acik'
			close_counter_r = 0
		else:
			state_r = 'Kapali'
			close_counter_r += 1

		if prediction_l > 0.5 :
			state_l = 'Acik'
			close_counter_l = 0
		else:
			state_l = 'Kapali'
			close_counter_l += 1


		face_count = 0
		for (x, y, w, h) in faces:
			face_count += 1

			if (face_count == 2):
				face_count = 0
				continue

			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
			roi_gray = gray[y:y + h, x: x + w]
			roi_color = frame[y:y + h, x: x + w]
			eyes = eye_cascade.detectMultiScale(roi_gray)
			eye_count = 0



			for (ex, ey, ew, eh) in eyes:

				eye_count += 1
				cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 0), 2)

				cv2.rectangle(frame, (210, 170), (380, 230), (0, 250, 255), 2)

				#cv2.putText(frame, str(x), (80, 35), font_type, 1, (0, 0, 255))
				#cv2.putText(frame, str(y), (80, 60), font_type, 1, (0, 0, 255))

				#fare kontrolü
				if (kalibre == 0):
					cv2.putText(frame, "kare icine gir", (100, 60), font_type, 1, (210, 0, 0))
					if (x < 170 and x > 140 and y > 80 and y < 100):
						pyautogui.moveTo(farex, farey)
						kalibre = 1
				else:
					if (x > 200 and y > 70 and y < 130):
						farex = farex + 5
					elif (x > 200 and y < 70):
						farex = farex + 5
						farey = farey - 5

					if (x > 220 and y > 60 and y < 150):
						farex = farex + 10
					elif (x > 220 and y < 60):
						farex = farex + 10
						farey = farey - 10

					if (x < 100 and y > 70 and y < 150):
						farex = farex - 5
					elif (x < 100 and y < 70):
						farex = farex - 5
						farey = farey - 5

					if (x < 80 and y > 60 and y < 150):
						farex = farex - 10
					elif (x < 80 and y < 50):
						farex = farex - 10
						farey = farey - 10

					if (x > 90 and y > 110 and x < 380):
						farey = farey + 5
					elif (y > 110 and x < 80):
						farey = farey + 5
						farex = farex - 5

					if (x > 90 and y > 130 and x < 380):
						farey = farey + 10
					elif (y > 130 and x < 90):
						farey = farey + 10
						farex = farex - 10

					if (x > 90 and y < 60 and x < 380):
						farey = farey - 5
					elif (x > 380 and y > 130):
						farey = farey + 5
						farex = farex + 5

					if (x > 90 and y < 80 and x < 380):
						farey = farey - 10
					elif (x > 390 and y>140):
						farey = farey + 10
						farex = farex + 10
					pyautogui.moveTo(farex, farey)


		if state_r == 'Acik' and mem_counter_r > 1:
			blinks_r += 1
			pyautogui.click()
            
		#sonraki döngü için sayacı tut
		mem_counter_r = close_counter_r


		if state_l == 'Acik' and mem_counter_l > 1:
			blinks_l += 1
			pyautogui.click(button='right')


		mem_counter_l = close_counter_l




		#toplam kırpmayı ekrana yazdır
		cv2.putText(frame, "Kirpma sag: {}".format(blinks_r), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Durum sag: {}".format(state_r), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Kirpma solx: {}".format(blinks_l), (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Durum sol: {}".format(state_l), (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		

		cv2.imshow('Fare kontrol', frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	del(camera)


if __name__ == '__main__':
	main()